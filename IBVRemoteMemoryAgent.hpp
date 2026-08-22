#ifndef IBV_REMOTE_MEMORY_AGENT_HPP
#define IBV_REMOTE_MEMORY_AGENT_HPP

#include <cassert>

#ifdef __WITH_IBV

#include "RemoteMemoryAgent.hpp"
#include "IBVContext.hpp"
#include <infiniband/verbs.h>
#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <memory>
#include <utility>

template<typename T>
class IBVRemoteMemoryAgent : public RemoteMemoryAgent<T>
{
public:
    IBVRemoteMemoryAgent(size_t count, IBVContext &context, MPI_Comm agent_comm)
        : count(count), context(context), agent_comm(agent_comm),
          owned_buffer(), buffer(nullptr), mr(nullptr),
          scratch(nullptr), scratch_mr(nullptr),
          staging_storage(), staging(nullptr), staging_mr(nullptr), staging_size(0),
          staging_next(0), staging_active_target(-1),
          freed(false), owns_memory(true)
    {
        this->BuildRankMap();
        this->context.EnsureConnected(this->rank_map, this->agent_comm);
        this->AllocateAndRegister(count);
        this->ExchangeRemoteInfo();
    }

    IBVRemoteMemoryAgent(T *user_buffer, size_t count, IBVContext &context, MPI_Comm agent_comm)
        : count(count), context(context), agent_comm(agent_comm),
          owned_buffer(), buffer(user_buffer), mr(nullptr),
          scratch(nullptr), scratch_mr(nullptr),
          staging_storage(), staging(nullptr), staging_mr(nullptr), staging_size(0),
          staging_next(0), staging_active_target(-1),
          freed(false), owns_memory(false)
    {
        this->BuildRankMap();
        this->context.EnsureConnected(this->rank_map, this->agent_comm);
        this->RegisterUserBuffer(count);
        this->ExchangeRemoteInfo();
    }

    ~IBVRemoteMemoryAgent() override
    {
        if(not std::uncaught_exceptions() and not this->freed)
        {
            this->Free();
        }
    }

    T *GetLocalPointer() override
    {
        return this->buffer;
    }

    size_t GetCount() const override
    {
        return this->count;
    }

    void Put(const T *origin, size_t count, int target_rank,
             size_t target_disp, bool flush = true, uint32_t source_lkey = 0) override
    {
        if(target_rank == this->my_agent_rank)
        {
            std::copy_n(origin, count, this->buffer + target_disp);
            return;
        }

        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];
        this->ValidateRemoteRange(remote, target_rank, target_disp, count, "Put");
        uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

        size_t payload_bytes = count * sizeof(T);
        const void *local_addr = origin;
        uint32_t local_lkey;
        ibv_mr *temp_mr = nullptr;

        if(source_lkey != 0)
        {
            local_lkey = source_lkey;
        }
        else if(this->IsInBuffer(origin, count))
        {
            local_lkey = this->BufferLkey();
        }
        else
        {
            if(payload_bytes <= this->context.GetMaxInlineData())
            {
                local_lkey = 0;
            }
            else if(payload_bytes >= DIRECT_REG_BYTE_THRESHOLD)
            {
                temp_mr = ibv_reg_mr(this->context.GetPD(), const_cast<T*>(origin), payload_bytes, IBV_ACCESS_LOCAL_WRITE);
                if(not temp_mr)
                {
                    throw std::runtime_error("IBVRemoteMemoryAgent::Put: ibv_reg_mr failed for direct source");
                }
                local_lkey = temp_mr->lkey;
            }
            else
            {
                T *staged = this->AllocateStaging(count, world_target);
                std::copy_n(origin, count, staged);
                local_addr = staged;
                local_lkey = this->StagingLkey();
            }
        }

        const bool signalWrite = flush or temp_mr;
        this->context.PostRDMAWrite(world_target, local_addr, payload_bytes, local_lkey, remote_addr, remote.rkey, signalWrite);

        if(flush or temp_mr)
        {
            this->context.DrainCompletions();
            this->ResetStaging();
        }
        if(temp_mr)
        {
            ibv_dereg_mr(temp_mr);
        }
    }

    void PutScatter(const T *contiguous_source, const uint32_t *target_disps, size_t count, int target_rank, bool flush = true) override
    {
        static std::vector<IBVContext::RDMAWriteEntry> entries;

        if(count == 0) return;

        if(target_rank == this->my_agent_rank)
        {
            for(size_t i = 0; i < count; i++)
            {
                this->buffer[target_disps[i]] = contiguous_source[i];
            }
            return;
        }

        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];

        size_t payload_bytes = count * sizeof(T);
        const T *local_source;
        uint32_t local_lkey;
        ibv_mr *temp_mr = nullptr;

        if(sizeof(T) <= this->context.GetMaxInlineData())
        {
            local_source = contiguous_source;
            local_lkey = 0;
        }
        else if(payload_bytes >= DIRECT_REG_BYTE_THRESHOLD)
        {
            temp_mr = ibv_reg_mr(this->context.GetPD(), const_cast<T*>(contiguous_source), payload_bytes, IBV_ACCESS_LOCAL_WRITE);
            if(not temp_mr)
            {
                throw std::runtime_error("IBVRemoteMemoryAgent::PutScatter: ibv_reg_mr failed for direct source");
            }
            local_source = contiguous_source;
            local_lkey = temp_mr->lkey;
        }
        else
        {
            T *staged = this->AllocateStaging(count, world_target);
            std::copy_n(contiguous_source, count, staged);
            local_source = staged;
            local_lkey = this->StagingLkey();
        }

        entries.resize(count);
        for(size_t i = 0; i < count; i++)
        {
            this->ValidateRemoteRange(remote, target_rank, target_disps[i], 1, "PutScatter");
            entries[i].local_addr = local_source + i;
            entries[i].bytes = static_cast<uint32_t>(sizeof(T));
            entries[i].remote_addr = remote.addr + target_disps[i] * sizeof(T);
        }

        const bool signalWrite = flush or temp_mr;
        this->context.PostRDMAWriteBatch(world_target, entries.data(), count, local_lkey, remote.rkey, signalWrite);

        if(flush or temp_mr)
        {
            this->context.DrainCompletions();
            this->ResetStaging();
        }
        if(temp_mr)
        {
            ibv_dereg_mr(temp_mr);
        }
    }

    void PutBatch(const T *source, size_t total_elements,
                  const typename RemoteMemoryAgent<T>::PutBatchEntry *entries, size_t num_entries,
                  int target_rank, bool flush = true, uint32_t source_lkey = 0) override
    {
        static std::vector<IBVContext::RDMAWriteEntry> rdma_entries;

        if(num_entries == 0) return;

        if(target_rank == this->my_agent_rank)
        {
            for(size_t i = 0; i < num_entries; i++)
            {
                std::copy_n(source + entries[i].source_offset, entries[i].count,
                            this->buffer + entries[i].target_disp);
            }
            return;
        }

        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];

        size_t payload_bytes = total_elements * sizeof(T);
        const T *local_source;
        uint32_t local_lkey;
        ibv_mr *temp_mr = nullptr;

        if(source_lkey != 0)
        {
            local_source = source;
            local_lkey = source_lkey;
        }
        else
        {
            size_t max_entry_bytes = 0;
            for(size_t i = 0; i < num_entries; i++)
            {
                size_t eb = entries[i].count * sizeof(T);
                if(eb > max_entry_bytes) max_entry_bytes = eb;
            }

            if(max_entry_bytes <= this->context.GetMaxInlineData())
            {
                local_source = source;
                local_lkey = 0;
            }
            else if(payload_bytes >= DIRECT_REG_BYTE_THRESHOLD)
            {
                temp_mr = ibv_reg_mr(this->context.GetPD(), const_cast<T*>(source), payload_bytes, IBV_ACCESS_LOCAL_WRITE);
                if(not temp_mr)
                {
                    throw std::runtime_error("IBVRemoteMemoryAgent::PutBatch: ibv_reg_mr failed for direct source");
                }
                local_source = source;
                local_lkey = temp_mr->lkey;
            }
            else
            {
                T *staged = this->AllocateStaging(total_elements, world_target);
                std::copy_n(source, total_elements, staged);
                local_source = staged;
                local_lkey = this->StagingLkey();
            }
        }

        rdma_entries.resize(num_entries);
        for(size_t i = 0; i < num_entries; i++)
        {
            this->ValidateRemoteRange(remote, target_rank, entries[i].target_disp,
                                      entries[i].count, "PutBatch");
            rdma_entries[i].local_addr = local_source + entries[i].source_offset;
            rdma_entries[i].bytes = static_cast<uint32_t>(entries[i].count * sizeof(T));
            rdma_entries[i].remote_addr = remote.addr + entries[i].target_disp * sizeof(T);
        }

        const bool signalWrite = flush or temp_mr;
        this->context.PostRDMAWriteBatch(world_target, rdma_entries.data(), num_entries, local_lkey, remote.rkey, signalWrite);

        if(flush or temp_mr)
        {
            this->context.DrainCompletions();
            this->ResetStaging();
        }
        if(temp_mr)
        {
            ibv_dereg_mr(temp_mr);
        }
    }

    typename RemoteMemoryAgent<T>::SourceRegistration RegisterExternalSource(const void *data, size_t bytes) override
    {
        if(bytes == 0) return {};
        ibv_mr *ext_mr = ibv_reg_mr(this->context.GetPD(), const_cast<void*>(data), bytes, IBV_ACCESS_LOCAL_WRITE);
        if(not ext_mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::RegisterExternalSource: ibv_reg_mr failed");
        }
        return {ext_mr->lkey, reinterpret_cast<uint64_t>(ext_mr)};
    }

    void DeregisterExternalSource(uint64_t handle) override
    {
        if(handle)
        {
            ibv_dereg_mr(reinterpret_cast<ibv_mr*>(handle));
        }
    }

    void Get(T *result, size_t count, int target_rank, size_t target_disp, bool flush = true) const override
    {
        if(target_rank == this->my_agent_rank)
        {
            std::copy_n(this->buffer + target_disp, count, result);
            return;
        }

        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];
        this->ValidateRemoteRange(remote, target_rank, target_disp, count, "Get");
        uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

        bool external = not this->IsInBuffer(result, count);
        void *local_addr = result;
        uint32_t local_lkey = this->BufferLkey();

        if(external)
        {
            local_addr = this->AllocateStaging(count, world_target);
            local_lkey = this->StagingLkey();
        }

        this->context.PostRDMARead(world_target, local_addr, count * sizeof(T), local_lkey, remote_addr, remote.rkey, true);

        if(flush)
        {
            this->context.DrainCompletions();
            if(external)
            {
                std::copy_n(static_cast<T*>(local_addr), count, result);
            }
            this->ResetStaging();
        }
    }

    void CompareAndSwap(const T &desired, const T &expected, T &old_value, int target_rank, size_t target_disp, bool flush = true) override
    {
        if constexpr(sizeof(T) <= 8)
        {
            int world_target = this->rank_map[target_rank];
            const IBVRemoteRegion &remote = this->remote_regions[target_rank];
            this->ValidateRemoteRange(remote, target_rank, target_disp, 1, "CompareAndSwap");
            uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

            uint64_t compare_val = 0, swap_val = 0;
            std::memcpy(&compare_val, &expected, sizeof(T));
            std::memcpy(&swap_val, &desired, sizeof(T));

            if constexpr(sizeof(T) == 4)
            {
                remote_addr = (remote_addr / 8) * 8;
            }

            this->context.PostAtomicCAS(world_target, this->scratch, this->ScratchLkey(), remote_addr, remote.rkey, compare_val, swap_val, true);

            if(flush)
            {
                this->context.DrainCompletions();
            }

            std::memcpy(&old_value, this->scratch, sizeof(T));
        }
        else
        {
            (void)desired; (void)expected; (void)old_value;
            (void)target_rank; (void)target_disp; (void)flush;
            throw std::runtime_error("CompareAndSwap requires sizeof(T) <= 8");
        }
    }

    T FetchAndAdd(const T &addend, int target_rank, size_t target_disp, bool flush = true) override
    {
        if constexpr(sizeof(T) <= 8)
        {
            int world_target = this->rank_map[target_rank];
            const IBVRemoteRegion &remote = this->remote_regions[target_rank];
            this->ValidateRemoteRange(remote, target_rank, target_disp, 1, "FetchAndAdd");
            uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

            uint64_t add_val = 0;
            std::memcpy(&add_val, &addend, sizeof(T));

            if constexpr(sizeof(T) == 4)
            {
                remote_addr = (remote_addr / 8) * 8;
            }

            this->context.PostAtomicFetchAdd(world_target, this->scratch, this->ScratchLkey(), remote_addr, remote.rkey, add_val, true);

            if(flush)
            {
                this->context.DrainCompletions();
            }

            T old_value;
            std::memcpy(&old_value, this->scratch, sizeof(T));
            return old_value;
        }
        else
        {
            (void)addend; (void)target_rank; (void)target_disp; (void)flush;
            throw std::runtime_error("FetchAndAdd requires sizeof(T) <= 8");
        }
    }

    void Flush(int target_rank) override
    {
        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];
        this->context.PostRDMARead(world_target, this->scratch, 1, this->ScratchLkey(), remote.addr, remote.rkey, true);
        this->context.DrainCompletions();
        this->ResetStaging();
    }

    void QuiesceTarget(int target_rank) override
    {
        if(target_rank == this->my_agent_rank)
        {
            return;
        }
        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];
        // Signaled read on the same RC QP orders after all prior writes/atomics.
        this->context.PostRDMARead(world_target, this->scratch, 1, this->ScratchLkey(),
                                   remote.addr, remote.rkey, true);
        this->context.DrainCompletionsForTarget(world_target);
        this->ResetStaging();
    }

    bool SupportsAsyncReallocation() const override
    {
        // Native verbs MRs become invalid immediately at ibv_dereg_mr. A
        // requester that still has an old address/rkey cached can therefore
        // turn an otherwise recoverable resize race into
        // IBV_WC_REM_ACCESS_ERR. Use STORM's pair-synchronized resize path
        // for IBV; LocalResize remains available as a primitive, but is not
        // advertised as safe for background/asynchronous replacement.
        return false;
    }

    bool SupportsPersistentSourceRegistration() const override
    {
        return true;
    }

    bool SupportsShrinkingReallocation() const override
    {
        // Keep verbs MRs monotonic. Repeated shrink/grow replacement churns
        // address/rkey generations and has produced remote protection errors
        // under CrookedPipe's highly imbalanced cycle-60 workload.
        return false;
    }

    void Resize(size_t new_count) override
    {
        if(not this->owns_memory)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::Resize: cannot resize user-supplied memory");
        }
        this->QuiesceAgentCommBeforeDeregister();

        size_t old_count = this->count;
        size_t copy_count = std::min(old_count, new_count);
        std::vector<T> saved;
        saved.reserve(copy_count);
        for(size_t i = 0; i < copy_count; i++)
        {
            saved.push_back(this->buffer[i]);
        }

        std::unique_ptr<T[]> new_storage =
            std::make_unique<T[]>(std::max<size_t>(new_count, 1));
        T *new_buffer = new_storage.get();
        for(size_t i = 0; i < copy_count; i++)
        {
            new_buffer[i] = saved[i];
        }

        ibv_mr *new_mr = ibv_reg_mr(this->context.GetPD(), new_buffer,
            std::max<size_t>(new_count, 1) * sizeof(T),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if(not new_mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::Resize: ibv_reg_mr failed");
        }

        if(this->mr)
        {
            ibv_dereg_mr(this->mr);
        }
        this->owned_buffer.reset();

        if(this->staging_mr)
        {
            ibv_dereg_mr(this->staging_mr);
            this->staging_mr = nullptr;
        }
        if(this->staging)
        {
            this->staging_storage.reset();
            this->staging = nullptr;
        }
        this->staging_size = 0;
        this->staging_next = 0;
        this->staging_active_target = -1;

        this->owned_buffer = std::move(new_storage);
        this->buffer = this->owned_buffer.get();
        this->mr = new_mr;
        this->count = new_count;

        this->ExchangeRemoteInfo();
    }

    bool SupportsLocalResize() const override
    {
        return true;
    }

    RemoteBufferInfo LocalResize(size_t new_count) override
    {
        if(not this->owns_memory)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::LocalResize: cannot resize user-supplied memory");
        }
        this->context.DrainCompletions();

        size_t old_count = this->count;
        size_t copy_count = std::min(old_count, new_count);

        // Save active data locally before releasing all MR resources.
        // Deregistering old MRs first avoids exhausting provider key slots.
        std::vector<T> saved;
        saved.reserve(copy_count);
        for(size_t i = 0; i < copy_count; i++)
        {
            saved.push_back(this->buffer[i]);
        }

        // Release staging, retired buffers, and old MR before registering new
        if(this->staging_mr)
        {
            ibv_dereg_mr(this->staging_mr);
            this->staging_mr = nullptr;
        }
        if(this->staging)
        {
            this->staging_storage.reset();
            this->staging = nullptr;
        }
        this->staging_size = 0;
        this->staging_next = 0;
        this->staging_active_target = -1;

        if(this->mr)
        {
            ibv_dereg_mr(this->mr);
            this->mr = nullptr;
        }
        if(this->buffer)
        {
            this->owned_buffer.reset();
            this->buffer = nullptr;
        }

        // Allocate and register new buffer with freed MR slots
        this->owned_buffer =
            std::make_unique<T[]>(std::max<size_t>(new_count, 1));
        T *new_buffer = this->owned_buffer.get();

        ibv_mr *new_mr = ibv_reg_mr(this->context.GetPD(), new_buffer,
            std::max<size_t>(new_count, 1) * sizeof(T),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if(not new_mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::LocalResize: ibv_reg_mr failed");
        }

        for(size_t i = 0; i < copy_count; i++)
        {
            new_buffer[i] = saved[i];
        }

        this->buffer = new_buffer;
        this->mr = new_mr;
        this->count = new_count;

        if(this->my_agent_rank >= 0 and this->my_agent_rank < static_cast<int>(this->remote_regions.size()))
        {
            this->remote_regions[this->my_agent_rank].addr = reinterpret_cast<uint64_t>(this->buffer);
            this->remote_regions[this->my_agent_rank].count = this->count;
            this->remote_regions[this->my_agent_rank].rkey = this->BufferRkey();
        }

        return this->GetLocalRemoteInfo();
    }

    RemoteBufferInfo GetLocalRemoteInfo() const override
    {
        RemoteBufferInfo info;
        info.addr = reinterpret_cast<uint64_t>(this->buffer);
        info.rkey = this->BufferRkey();
        info.count = this->count;
        return info;
    }

    void UpdateRemoteInfo(int peer_rank, const RemoteBufferInfo &info) override
    {
        if(peer_rank < 0 or peer_rank >= static_cast<int>(this->remote_regions.size()))
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::UpdateRemoteInfo: peer rank is out of range");
        }
        this->remote_regions[peer_rank].addr = info.addr;
        this->remote_regions[peer_rank].count = info.count;
        this->remote_regions[peer_rank].rkey = static_cast<uint32_t>(info.rkey);
    }

    void Replace(size_t new_count) override
    {
        if(not this->owns_memory)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::Replace: cannot replace user-supplied memory");
        }
        this->QuiesceAgentCommBeforeDeregister();

        if(this->staging_mr)
        {
            ibv_dereg_mr(this->staging_mr);
            this->staging_mr = nullptr;
        }
        if(this->staging)
        {
            this->staging_storage.reset();
            this->staging = nullptr;
        }
        this->staging_size = 0;
        this->staging_next = 0;
        this->staging_active_target = -1;

        if(this->mr)
        {
            ibv_dereg_mr(this->mr);
            this->mr = nullptr;
        }
        if(this->buffer)
        {
            this->owned_buffer.reset();
            this->buffer = nullptr;
        }
        this->count = 0;

        this->owned_buffer =
            std::make_unique<T[]>(std::max<size_t>(new_count, 1));
        this->buffer = this->owned_buffer.get();

        this->mr = ibv_reg_mr(this->context.GetPD(), this->buffer,
            std::max<size_t>(new_count, 1) * sizeof(T),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if(not this->mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::Replace: ibv_reg_mr failed");
        }

        this->count = new_count;
        this->ExchangeRemoteInfo();
    }

    void Free() override
    {
        if(this->freed)
        {
            return;
        }
        this->context.DrainCompletions();

        if(this->staging_mr)
        {
            ibv_dereg_mr(this->staging_mr);
            this->staging_mr = nullptr;
        }
        if(this->staging)
        {
            this->staging_storage.reset();
            this->staging = nullptr;
        }
        if(this->scratch_mr)
        {
            ibv_dereg_mr(this->scratch_mr);
            this->scratch_mr = nullptr;
        }
        if(this->scratch)
        {
            rma_detail::advise_dontneed(this->scratch, sizeof(T));
            std::free(this->scratch);
            this->scratch = nullptr;
        }
        if(this->mr)
        {
            ibv_dereg_mr(this->mr);
            this->mr = nullptr;
        }
        if(this->buffer)
        {
            if(this->owns_memory)
            {
                this->owned_buffer.reset();
            }
            this->buffer = nullptr;
        }

        this->count = 0;
        this->staging_size = 0;
        this->staging_next = 0;
        this->staging_active_target = -1;
        this->freed = true;
    }

    static std::unique_ptr<IBVRemoteMemoryAgent<T>> Create(size_t count, IBVContext &context, MPI_Comm agent_comm)
    {
        return std::make_unique<IBVRemoteMemoryAgent<T>>(count, context, agent_comm);
    }

private:
    size_t count;
    IBVContext &context;
    MPI_Comm agent_comm;
    int my_agent_rank;
    std::vector<int> rank_map;
    std::unique_ptr<T[]> owned_buffer;
    T *buffer;
    ibv_mr *mr;
    uint64_t *scratch;
    ibv_mr *scratch_mr;
    mutable std::unique_ptr<T[]> staging_storage;
    mutable T *staging;
    mutable ibv_mr *staging_mr;
    mutable size_t staging_size;
    mutable size_t staging_next;
    mutable int staging_active_target;
    std::vector<IBVRemoteRegion> remote_regions;
    bool freed;
    bool owns_memory;

    static constexpr size_t DIRECT_REG_BYTE_THRESHOLD = 8192;

    void ValidateRemoteRange(const IBVRemoteRegion &remote, int target_rank,
                             size_t target_disp, size_t transfer_count,
                             const char *operation) const
    {
        if(target_disp <= remote.count and transfer_count <= remote.count - target_disp)
        {
            return;
        }

        int world_target = -1;
        if(target_rank >= 0 and target_rank < static_cast<int>(this->rank_map.size()))
        {
            world_target = this->rank_map[target_rank];
        }
        throw std::runtime_error(
            std::string("IBVRemoteMemoryAgent::") + operation +
            ": remote range exceeds registered MR"
            " target_rank=" + std::to_string(target_rank) +
            " world_target=" + std::to_string(world_target) +
            " target_disp=" + std::to_string(target_disp) +
            " transfer_count=" + std::to_string(transfer_count) +
            " remote_count=" + std::to_string(remote.count) +
            " remote_addr=" + std::to_string(remote.addr) +
            " remote_rkey=" + std::to_string(remote.rkey));
    }

    uint32_t BufferLkey() const {return this->mr->lkey;}
    uint32_t ScratchLkey() const {return this->scratch_mr->lkey;}
    uint32_t StagingLkey() const {return this->staging_mr->lkey;}
    uint32_t BufferRkey() const {return this->mr->rkey;}

    void ResetStaging() const
    {
        this->staging_next = 0;
        this->staging_active_target = -1;
    }

    T *AllocateStaging(size_t required_count, int world_target) const
    {
        if(required_count == 0)
        {
            required_count = 1;
        }

        if(this->staging_active_target == -1)
        {
            this->staging_active_target = world_target;
        }
        else if(this->staging_active_target != world_target)
        {
            this->staging_active_target = -2;
        }

        if(required_count > this->staging_size)
        {
            this->EnsureStaging(required_count);
            this->staging_active_target = world_target;
        }

        if(this->staging_next + required_count > this->staging_size)
        {
            this->context.DrainCompletions();
            this->ResetStaging();
            this->staging_active_target = world_target;
            if(required_count > this->staging_size)
            {
                this->EnsureStaging(required_count);
            }
        }

        T *result = this->staging + this->staging_next;
        this->staging_next += required_count;
        return result;
    }

    void BuildRankMap()
    {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        int agent_size;
        MPI_Comm_size(this->agent_comm, &agent_size);
        MPI_Comm_rank(this->agent_comm, &this->my_agent_rank);
        this->rank_map.resize(agent_size);
        MPI_Allgather(&world_rank, 1, MPI_INT, this->rank_map.data(), 1, MPI_INT, this->agent_comm);
    }

    // Inbound RDMA writes do not generate CQEs on this rank. DrainCompletions
    // only waits for our outbound signaled WRs, so a peer can still be writing
    // into this MR. Fence every peer QP, then barrier, then it is safe to
    // ibv_dereg_mr.
    void QuiesceAgentCommBeforeDeregister()
    {
        this->context.DrainCompletions();
        for(size_t i = 0; i < this->remote_regions.size(); ++i)
        {
            if(static_cast<int>(i) == this->my_agent_rank)
            {
                continue;
            }
            const IBVRemoteRegion &remote = this->remote_regions[i];
            if(remote.addr == 0)
            {
                continue;
            }
            int world_target = this->rank_map[i];
            this->context.PostRDMARead(world_target, this->scratch, 1, this->ScratchLkey(),
                                       remote.addr, remote.rkey, true);
        }
        this->context.DrainCompletions();
        MPI_Barrier(this->agent_comm);
    }

    bool IsInBuffer(const T *ptr, size_t n) const
    {
        auto buf_begin = reinterpret_cast<uintptr_t>(this->buffer);
        auto buf_end = buf_begin + this->count * sizeof(T);
        auto p_begin = reinterpret_cast<uintptr_t>(ptr);
        auto p_end = p_begin + n * sizeof(T);
        return (p_begin >= buf_begin and p_end <= buf_end);
    }

    void EnsureStaging(size_t required_count) const
    {
        if(this->staging and this->staging_size >= required_count)
        {
            return;
        }
        if(this->staging_mr)
        {
            this->context.DrainCompletions();
            ibv_dereg_mr(this->staging_mr);
        }
        if(this->staging)
        {
            this->staging_storage.reset();
            this->staging = nullptr;
        }

        size_t new_size = std::max(required_count, this->count);
        if(new_size == 0) new_size = 1;
        if(this->staging_size > 0)
        {
            new_size = std::max(new_size, this->staging_size * 2);
        }
        this->staging_storage = std::make_unique<T[]>(new_size);
        this->staging = this->staging_storage.get();

        this->staging_mr = ibv_reg_mr(this->context.GetPD(), this->staging,
            new_size * sizeof(T), IBV_ACCESS_LOCAL_WRITE);
        if(not this->staging_mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: ibv_reg_mr failed for staging");
        }
        this->staging_size = new_size;
        this->staging_next = 0;
    }

    void RegisterUserBuffer(size_t count)
    {
        size_t reg_size = count * sizeof(T);
        if(reg_size == 0) reg_size = sizeof(T);

        this->mr = ibv_reg_mr(this->context.GetPD(), this->buffer, reg_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if(not this->mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: ibv_reg_mr failed for user buffer");
        }

        this->scratch = static_cast<uint64_t*>(std::aligned_alloc(64, 64));
        if(not this->scratch)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: aligned_alloc failed for scratch");
        }

        this->scratch_mr = ibv_reg_mr(this->context.GetPD(), this->scratch, 64, IBV_ACCESS_LOCAL_WRITE);
        if(not this->scratch_mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: ibv_reg_mr failed for scratch");
        }
    }

    void AllocateAndRegister(size_t count)
    {
        this->owned_buffer =
            std::make_unique<T[]>(std::max<size_t>(count, 1));
        this->buffer = this->owned_buffer.get();

        this->mr = ibv_reg_mr(this->context.GetPD(), this->buffer,
            std::max<size_t>(count, 1) * sizeof(T),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if(not this->mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: ibv_reg_mr failed for buffer");
        }

        this->scratch = static_cast<uint64_t*>(std::aligned_alloc(64, 64));
        if(not this->scratch)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: aligned_alloc failed for scratch");
        }

        this->scratch_mr = ibv_reg_mr(this->context.GetPD(), this->scratch, 64, IBV_ACCESS_LOCAL_WRITE);
        if(not this->scratch_mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: ibv_reg_mr failed for scratch");
        }
    }

    void ExchangeRemoteInfo()
    {
        IBVRemoteRegion local_info{};
        local_info.addr = reinterpret_cast<uint64_t>(this->buffer);
        local_info.count = this->count;
        local_info.rkey = this->BufferRkey();

        int size;
        MPI_Comm_size(this->agent_comm, &size);
        this->remote_regions.resize(size);

        MPI_Allgather(&local_info, sizeof(IBVRemoteRegion), MPI_BYTE, this->remote_regions.data(), sizeof(IBVRemoteRegion), MPI_BYTE, this->agent_comm);
    }
};

#endif // __WITH_IBV

#endif // IBV_REMOTE_MEMORY_AGENT_HPP
