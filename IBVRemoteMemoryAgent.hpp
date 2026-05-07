#ifndef IBV_REMOTE_MEMORY_AGENT_HPP
#define IBV_REMOTE_MEMORY_AGENT_HPP

#ifdef __WITH_IBV

#include "RemoteMemoryAgent.hpp"
#include "IBVContext.hpp"
#include <infiniband/verbs.h>
#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <memory>

template<typename T>
class IBVRemoteMemoryAgent : public RemoteMemoryAgent<T>
{
public:
    IBVRemoteMemoryAgent(size_t count, IBVContext &context, MPI_Comm agent_comm)
        : count(count), context(context), agent_comm(agent_comm),
          buffer(nullptr), mr(nullptr),
          scratch(nullptr), scratch_mr(nullptr),
          staging(nullptr), staging_mr(nullptr), staging_size(0),
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
          buffer(user_buffer), mr(nullptr),
          scratch(nullptr), scratch_mr(nullptr),
          staging(nullptr), staging_mr(nullptr), staging_size(0),
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
             size_t target_disp, bool flush = true) override
    {
        if(target_rank == this->my_agent_rank)
        {
            std::memcpy(this->buffer + target_disp, origin, count * sizeof(T));
            return;
        }

        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];
        uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

        size_t payload_bytes = count * sizeof(T);
        const void *local_addr = origin;
        uint32_t local_lkey = this->BufferLkey();
        ibv_mr *temp_mr = nullptr;

        if(not this->IsInBuffer(origin, count))
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
                std::memcpy(staged, origin, payload_bytes);
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
            std::memcpy(staged, contiguous_source, payload_bytes);
            local_source = staged;
            local_lkey = this->StagingLkey();
        }

        entries.resize(count);
        for(size_t i = 0; i < count; i++)
        {
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

    void Get(T *result, size_t count, int target_rank, size_t target_disp, bool flush = true) const override
    {
        if(target_rank == this->my_agent_rank)
        {
            std::memcpy(result, this->buffer + target_disp, count * sizeof(T));
            return;
        }

        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];
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
                std::memcpy(result, local_addr, count * sizeof(T));
            }
            this->ResetStaging();
        }
    }

    void CompareAndSwap(const T &desired, const T &expected, T &old_value, int target_rank, size_t target_disp, bool flush = true) override
    {
        assert(sizeof(T) == 4 or sizeof(T) == 8);

        // Always use RDMA CAS (including loopback for local target)
        // to ensure atomicity with remote RDMA CAS from other ranks.
        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];
        uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

        uint64_t compare_val = 0, swap_val = 0;
        std::memcpy(&compare_val, &expected, sizeof(T));
        std::memcpy(&swap_val, &desired, sizeof(T));

        if(sizeof(T) == 4)
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

    T FetchAndAdd(const T &addend, int target_rank, size_t target_disp, bool flush = true) override
    {
        assert(sizeof(T) == 4 or sizeof(T) == 8);

        // Always use RDMA FAA (including loopback for local target)
        // to ensure atomicity with remote RDMA FAA from other ranks.
        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];
        uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

        uint64_t add_val = 0;
        std::memcpy(&add_val, &addend, sizeof(T));

        if(sizeof(T) == 4)
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

    void Flush(int target_rank) override
    {
        int world_target = this->rank_map[target_rank];
        const IBVRemoteRegion &remote = this->remote_regions[target_rank];
        this->context.PostRDMARead(world_target, this->scratch, 1, this->ScratchLkey(), remote.addr, remote.rkey, true);
        this->context.DrainCompletions();
        this->ResetStaging();
    }

    void Resize(size_t new_count) override
    {
        if(not this->owns_memory)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::Resize: cannot resize user-supplied memory");
        }
        this->context.DrainCompletions();

        T *old_buffer = this->buffer;
        size_t old_count = this->count;

        size_t new_alloc_size = new_count * sizeof(T);
        if(new_alloc_size == 0) new_alloc_size = sizeof(T);
        size_t new_aligned_size = ((new_alloc_size + 63) / 64) * 64;

        T *new_buffer = static_cast<T*>(std::aligned_alloc(64, new_aligned_size));
        if(not new_buffer and new_count > 0)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::Resize: aligned_alloc failed");
        }
        std::memset(new_buffer, 0, new_aligned_size);

        ibv_mr *new_mr = ibv_reg_mr(this->context.GetPD(), new_buffer, new_aligned_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if(not new_mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::Resize: ibv_reg_mr failed");
        }

        size_t copy_count = std::min(old_count, new_count);
        if(copy_count > 0)
        {
            std::memcpy(new_buffer, old_buffer, copy_count * sizeof(T));
        }

        if(this->mr)
        {
            ibv_dereg_mr(this->mr);
        }
        rma_detail::advise_dontneed(old_buffer, old_count * sizeof(T));
        std::free(old_buffer);

        if(this->staging_mr)
        {
            ibv_dereg_mr(this->staging_mr);
            this->staging_mr = nullptr;
        }
        if(this->staging)
        {
            rma_detail::advise_dontneed(this->staging, this->staging_size * sizeof(T));
            std::free(this->staging);
            this->staging = nullptr;
        }
        this->staging_size = 0;
        this->staging_next = 0;
        this->staging_active_target = -1;

        this->buffer = new_buffer;
        this->mr = new_mr;
        this->count = new_count;

        this->ExchangeRemoteInfo();
    }

    void Replace(size_t new_count) override
    {
        if(not this->owns_memory)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::Replace: cannot replace user-supplied memory");
        }
        this->context.DrainCompletions();

        if(this->staging_mr)
        {
            ibv_dereg_mr(this->staging_mr);
            this->staging_mr = nullptr;
        }
        if(this->staging)
        {
            rma_detail::advise_dontneed(this->staging, this->staging_size * sizeof(T));
            std::free(this->staging);
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
            rma_detail::advise_dontneed(this->buffer, this->count * sizeof(T));
            std::free(this->buffer);
            this->buffer = nullptr;
        }
        this->count = 0;

        size_t new_alloc_size = new_count * sizeof(T);
        if(new_alloc_size == 0) new_alloc_size = sizeof(T);
        size_t new_aligned_size = ((new_alloc_size + 63) / 64) * 64;

        this->buffer = static_cast<T*>(std::aligned_alloc(64, new_aligned_size));
        if(not this->buffer and new_count > 0)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent::Replace: aligned_alloc failed");
        }
        std::memset(this->buffer, 0, new_aligned_size);

        this->mr = ibv_reg_mr(this->context.GetPD(), this->buffer, new_aligned_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
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
            rma_detail::advise_dontneed(this->staging, this->staging_size * sizeof(T));
            std::free(this->staging);
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
                rma_detail::advise_dontneed(this->buffer, this->count * sizeof(T));
                std::free(this->buffer);
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
    T *buffer;
    ibv_mr *mr;
    uint64_t *scratch;
    ibv_mr *scratch_mr;
    mutable T *staging;
    mutable ibv_mr *staging_mr;
    mutable size_t staging_size;
    mutable size_t staging_next;
    mutable int staging_active_target;
    std::vector<IBVRemoteRegion> remote_regions;
    bool freed;
    bool owns_memory;

    static constexpr size_t DIRECT_REG_BYTE_THRESHOLD = 8192;

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
            std::free(this->staging);
        }

        size_t new_size = std::max(required_count, this->count);
        if(new_size == 0) new_size = 1;
        if(this->staging_size > 0)
        {
            new_size = std::max(new_size, this->staging_size * 2);
        }
        size_t alloc_bytes = new_size * sizeof(T);
        size_t aligned_bytes = ((alloc_bytes + 63) / 64) * 64;

        this->staging = static_cast<T*>(std::aligned_alloc(64, aligned_bytes));
        if(not this->staging)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: aligned_alloc failed for staging");
        }
        std::memset(this->staging, 0, aligned_bytes);

        this->staging_mr = ibv_reg_mr(this->context.GetPD(), this->staging, aligned_bytes, IBV_ACCESS_LOCAL_WRITE);
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

        this->scratch_mr = ibv_reg_mr(this->context.GetPD(), this->scratch, 64, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if(not this->scratch_mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: ibv_reg_mr failed for scratch");
        }
    }

    void AllocateAndRegister(size_t count)
    {
        size_t alloc_size = count * sizeof(T);
        if(alloc_size == 0)
        {
            alloc_size = sizeof(T);
        }

        size_t aligned_size = ((alloc_size + 63) / 64) * 64;
        this->buffer = static_cast<T*>(std::aligned_alloc(64, aligned_size));
        if(not this->buffer)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: aligned_alloc failed for buffer");
        }
        std::memset(this->buffer, 0, aligned_size);

        this->mr = ibv_reg_mr(this->context.GetPD(), this->buffer, aligned_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if(not this->mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: ibv_reg_mr failed for buffer");
        }

        this->scratch = static_cast<uint64_t*>(std::aligned_alloc(64, 64));
        if(not this->scratch)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: aligned_alloc failed for scratch");
        }

        this->scratch_mr = ibv_reg_mr(this->context.GetPD(), this->scratch, 64, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        if(not this->scratch_mr)
        {
            throw std::runtime_error("IBVRemoteMemoryAgent: ibv_reg_mr failed for scratch");
        }
    }

    void ExchangeRemoteInfo()
    {
        IBVRemoteRegion local_info{};
        local_info.addr = reinterpret_cast<uint64_t>(this->buffer);
        local_info.rkey = this->BufferRkey();

        int size;
        MPI_Comm_size(this->agent_comm, &size);
        this->remote_regions.resize(size);

        MPI_Allgather(&local_info, sizeof(IBVRemoteRegion), MPI_BYTE, this->remote_regions.data(), sizeof(IBVRemoteRegion), MPI_BYTE, this->agent_comm);
    }
};

#endif // __WITH_IBV

#endif // IBV_REMOTE_MEMORY_AGENT_HPP
