#ifndef OFI_REMOTE_MEMORY_AGENT_HPP
#define OFI_REMOTE_MEMORY_AGENT_HPP

#include <cassert>

#ifdef __WITH_OFI

#include "RemoteMemoryAgent.hpp"
#include "OFIContext.hpp"
#include <rdma/fabric.h>
#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <memory>

template<typename T>
class OFIRemoteMemoryAgent : public RemoteMemoryAgent<T>
{
public:
    OFIRemoteMemoryAgent(size_t count, OFIContext &context, MPI_Comm agent_comm)
        : count(count), context(context), agent_comm(agent_comm),
          buffer(nullptr), mr(nullptr),
          scratch(nullptr), scratch_mr(nullptr),
          staging(nullptr), staging_mr(nullptr), staging_size(0),
          staging_next(0), staging_active_target(-1),
          next_ext_key(1), freed(false), owns_memory(true)
    {
        this->BuildRankMap();
        this->context.EnsureConnected(this->rank_map, this->agent_comm);
        this->AllocateAndRegister(count);
        this->ExchangeRemoteInfo();
    }

    OFIRemoteMemoryAgent(T *user_buffer, size_t count, OFIContext &context, MPI_Comm agent_comm)
        : count(count), context(context), agent_comm(agent_comm),
          buffer(user_buffer), mr(nullptr),
          scratch(nullptr), scratch_mr(nullptr),
          staging(nullptr), staging_mr(nullptr), staging_size(0),
          staging_next(0), staging_active_target(-1),
          next_ext_key(1), freed(false), owns_memory(false)
    {
        this->BuildRankMap();
        this->context.EnsureConnected(this->rank_map, this->agent_comm);
        this->RegisterUserBuffer(count);
        this->ExchangeRemoteInfo();
    }

    ~OFIRemoteMemoryAgent() override
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
            std::memcpy(this->buffer + target_disp, origin, count * sizeof(T));
            return;
        }

        int world_target = this->rank_map[target_rank];
        const OFIRemoteRegion &remote = this->remote_regions[target_rank];
        uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

        size_t payload_bytes = count * sizeof(T);
        const void *local_addr = origin;
        void *local_desc = nullptr;
        fid_mr *temp_mr = nullptr;

        if(source_lkey != 0)
        {
            auto it = this->external_registrations.find(source_lkey);
            if(it == this->external_registrations.end())
            {
                throw std::runtime_error("OFIRemoteMemoryAgent::Put: unknown source_lkey");
            }
            local_desc = it->second.desc;
        }
        else if(this->IsInBuffer(origin, count))
        {
            local_desc = this->BufferDesc();
        }
        else
        {
            if(payload_bytes >= DIRECT_REG_BYTE_THRESHOLD)
            {
                temp_mr = this->context.RegisterMemory(const_cast<T*>(origin), payload_bytes,
                    FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
                if(not temp_mr)
                {
                    throw std::runtime_error("OFIRemoteMemoryAgent::Put: fi_mr_reg failed for direct source");
                }
                local_desc = fi_mr_desc(temp_mr);
            }
            else
            {
                T *staged = this->AllocateStaging(count, world_target);
                std::memcpy(staged, origin, payload_bytes);
                local_addr = staged;
                local_desc = this->StagingDesc();
            }
        }

        const bool signalWrite = flush or temp_mr;
        this->context.PostRDMAWrite(world_target, local_addr, payload_bytes,
                                    local_desc, remote_addr, remote.key, signalWrite);

        if(flush or temp_mr)
        {
            this->context.DrainCompletions();
            this->ResetStaging();
        }
        if(temp_mr)
        {
            this->context.DeregisterMemory(temp_mr);
        }
    }

    void PutScatter(const T *contiguous_source, const uint32_t *target_disps,
                    size_t count, int target_rank, bool flush = true) override
    {
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
        const OFIRemoteRegion &remote = this->remote_regions[target_rank];

        size_t payload_bytes = count * sizeof(T);
        const T *local_source;
        void *local_desc = nullptr;
        fid_mr *temp_mr = nullptr;

        if(payload_bytes >= DIRECT_REG_BYTE_THRESHOLD)
        {
            temp_mr = this->context.RegisterMemory(const_cast<T*>(contiguous_source), payload_bytes,
                FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
            if(not temp_mr)
            {
                throw std::runtime_error("OFIRemoteMemoryAgent::PutScatter: fi_mr_reg failed for direct source");
            }
            local_source = contiguous_source;
            local_desc = fi_mr_desc(temp_mr);
        }
        else
        {
            T *staged = this->AllocateStaging(count, world_target);
            std::memcpy(staged, contiguous_source, payload_bytes);
            local_source = staged;
            local_desc = this->StagingDesc();
        }

        for(size_t i = 0; i < count; i++)
        {
            uint64_t remote_addr = remote.addr + target_disps[i] * sizeof(T);
            this->context.PostRDMAWrite(world_target, local_source + i, sizeof(T),
                                        local_desc, remote_addr, remote.key, false);
        }

        if(flush or temp_mr)
        {
            this->context.DrainCompletions();
            this->ResetStaging();
        }
        if(temp_mr)
        {
            this->context.DeregisterMemory(temp_mr);
        }
    }

    void PutBatch(const T *source, size_t total_elements,
                  const typename RemoteMemoryAgent<T>::PutBatchEntry *entries, size_t num_entries,
                  int target_rank, bool flush = true, uint32_t source_lkey = 0) override
    {
        if(num_entries == 0) return;

        if(target_rank == this->my_agent_rank)
        {
            for(size_t i = 0; i < num_entries; i++)
            {
                std::memcpy(this->buffer + entries[i].target_disp,
                            source + entries[i].source_offset,
                            entries[i].count * sizeof(T));
            }
            return;
        }

        int world_target = this->rank_map[target_rank];
        const OFIRemoteRegion &remote = this->remote_regions[target_rank];

        size_t payload_bytes = total_elements * sizeof(T);
        const T *local_source;
        void *local_desc = nullptr;
        fid_mr *temp_mr = nullptr;

        if(source_lkey != 0)
        {
            auto it = this->external_registrations.find(source_lkey);
            if(it == this->external_registrations.end())
            {
                throw std::runtime_error("OFIRemoteMemoryAgent::PutBatch: unknown source_lkey");
            }
            local_source = source;
            local_desc = it->second.desc;
        }
        else
        {
            if(payload_bytes >= DIRECT_REG_BYTE_THRESHOLD)
            {
                temp_mr = this->context.RegisterMemory(const_cast<T*>(source), payload_bytes,
                    FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
                if(not temp_mr)
                {
                    throw std::runtime_error("OFIRemoteMemoryAgent::PutBatch: fi_mr_reg failed for direct source");
                }
                local_source = source;
                local_desc = fi_mr_desc(temp_mr);
            }
            else
            {
                T *staged = this->AllocateStaging(total_elements, world_target);
                std::memcpy(staged, source, payload_bytes);
                local_source = staged;
                local_desc = this->StagingDesc();
            }
        }

        for(size_t i = 0; i < num_entries; i++)
        {
            uint64_t remote_addr = remote.addr + entries[i].target_disp * sizeof(T);
            this->context.PostRDMAWrite(world_target, local_source + entries[i].source_offset,
                entries[i].count * sizeof(T), local_desc, remote_addr, remote.key, false);
        }

        if(flush or temp_mr)
        {
            this->context.DrainCompletions();
            this->ResetStaging();
        }
        if(temp_mr)
        {
            this->context.DeregisterMemory(temp_mr);
        }
    }

    typename RemoteMemoryAgent<T>::SourceRegistration RegisterExternalSource(const void *data, size_t bytes) override
    {
        if(bytes == 0) return {};
        fid_mr *ext_mr = this->context.RegisterMemory(const_cast<void*>(data), bytes,
            FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
        if(not ext_mr)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent::RegisterExternalSource: fi_mr_reg failed");
        }
        uint32_t key = this->next_ext_key++;
        this->external_registrations[key] = {ext_mr, fi_mr_desc(ext_mr)};
        return {key, reinterpret_cast<uint64_t>(ext_mr)};
    }

    void DeregisterExternalSource(uint64_t handle) override
    {
        if(not handle) return;
        fid_mr *ext_mr = reinterpret_cast<fid_mr*>(handle);
        for(auto it = this->external_registrations.begin(); it != this->external_registrations.end(); ++it)
        {
            if(it->second.mr == ext_mr)
            {
                this->external_registrations.erase(it);
                break;
            }
        }
        this->context.DeregisterMemory(ext_mr);
    }

    void Get(T *result, size_t count, int target_rank, size_t target_disp, bool flush = true) const override
    {
        if(target_rank == this->my_agent_rank)
        {
            std::memcpy(result, this->buffer + target_disp, count * sizeof(T));
            return;
        }

        int world_target = this->rank_map[target_rank];
        const OFIRemoteRegion &remote = this->remote_regions[target_rank];
        uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

        bool external = not this->IsInBuffer(result, count);
        void *local_addr = result;
        void *local_desc = this->BufferDesc();

        if(external)
        {
            local_addr = this->AllocateStaging(count, world_target);
            local_desc = this->StagingDesc();
        }

        this->context.PostRDMARead(world_target, local_addr, count * sizeof(T),
                                   local_desc, remote_addr, remote.key, true);

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

    void CompareAndSwap(const T &desired, const T &expected, T &old_value,
                        int target_rank, size_t target_disp, bool flush = true) override
    {
        if constexpr(sizeof(T) <= 8)
        {
            int world_target = this->rank_map[target_rank];
            const OFIRemoteRegion &remote = this->remote_regions[target_rank];
            uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

            // scratch layout: [0]=result, [1]=swap, [2]=compare
            // All operands must reside in registered memory for fi_compare_atomic.
            this->scratch[1] = 0;
            this->scratch[2] = 0;
            std::memcpy(&this->scratch[1], &desired, sizeof(T));
            std::memcpy(&this->scratch[2], &expected, sizeof(T));

            fi_datatype dtype = (sizeof(T) <= 4) ? FI_UINT32 : FI_UINT64;

            if constexpr(sizeof(T) == 4)
            {
                remote_addr = (remote_addr / 8) * 8;
            }

            this->context.PostAtomicCAS(world_target,
                &this->scratch[1], this->ScratchDesc(),
                &this->scratch[2], this->ScratchDesc(),
                &this->scratch[0], this->ScratchDesc(),
                remote_addr, remote.key, dtype, true);

            if(flush)
            {
                this->context.DrainCompletions();
            }

            std::memcpy(&old_value, &this->scratch[0], sizeof(T));
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
            const OFIRemoteRegion &remote = this->remote_regions[target_rank];
            uint64_t remote_addr = remote.addr + target_disp * sizeof(T);

            // scratch layout: [0]=result, [1]=addend
            // All operands must reside in registered memory for fi_fetch_atomic.
            this->scratch[1] = 0;
            std::memcpy(&this->scratch[1], &addend, sizeof(T));

            fi_datatype dtype = (sizeof(T) <= 4) ? FI_UINT32 : FI_UINT64;

            if constexpr(sizeof(T) == 4)
            {
                remote_addr = (remote_addr / 8) * 8;
            }

            this->context.PostAtomicFetchAdd(world_target,
                &this->scratch[1], this->ScratchDesc(),
                &this->scratch[0], this->ScratchDesc(),
                remote_addr, remote.key, dtype, true);

            if(flush)
            {
                this->context.DrainCompletions();
            }

            T old_value;
            std::memcpy(&old_value, &this->scratch[0], sizeof(T));
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
        const OFIRemoteRegion &remote = this->remote_regions[target_rank];
        this->context.PostRDMARead(world_target, this->scratch, 1,
                                   this->ScratchDesc(), remote.addr, remote.key, true);
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
        const OFIRemoteRegion &remote = this->remote_regions[target_rank];
        // Fenced read: FI_FENCE orders this after all prior transmit operations,
        // and the read's round-trip guarantees remote visibility of prior writes.
        this->context.PostFencedRDMARead(world_target, this->scratch, 1,
                                         this->ScratchDesc(), remote.addr, remote.key);
        this->context.DrainCompletions();
        this->ResetStaging();
    }

    bool SupportsAsyncReallocation() const override
    {
        return this->SupportsLocalResize();
    }

    void MakeProgress() override
    {
        this->context.MakeProgress();
    }

    void Resize(size_t new_count) override
    {
        if(not this->owns_memory)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent::Resize: cannot resize user-supplied memory");
        }
        this->context.DrainCompletions();

        size_t old_count = this->count;
        size_t copy_count = std::min(old_count, new_count);

        std::vector<unsigned char> saved;
        if(copy_count > 0)
        {
            saved.resize(copy_count * sizeof(T));
            std::memcpy(saved.data(), this->buffer, saved.size());
        }

        this->ResetStaging();

        if(this->mr)
        {
            this->context.DeregisterMemory(this->mr);
            this->mr = nullptr;
        }
        if(this->buffer)
        {
            rma_detail::advise_dontneed(this->buffer, old_count * sizeof(T));
            std::free(this->buffer);
            this->buffer = nullptr;
        }

        size_t new_alloc_size = new_count * sizeof(T);
        if(new_alloc_size == 0) new_alloc_size = sizeof(T);
        size_t new_aligned_size = ((new_alloc_size + 63) / 64) * 64;

        T *new_buffer = static_cast<T*>(std::aligned_alloc(64, new_aligned_size));
        if(not new_buffer)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent::Resize: aligned_alloc failed");
        }
        std::memset(new_buffer, 0, new_aligned_size);

        fid_mr *new_mr = this->context.RegisterMemory(new_buffer, new_aligned_size,
            FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
        if(not new_mr)
        {
            std::free(new_buffer);
            throw std::runtime_error("OFIRemoteMemoryAgent::Resize: fi_mr_reg failed");
        }

        if(copy_count > 0)
        {
            std::memcpy(new_buffer, saved.data(), saved.size());
        }

        this->buffer = new_buffer;
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
            throw std::runtime_error("OFIRemoteMemoryAgent::LocalResize: cannot resize user-supplied memory");
        }
        this->context.DrainCompletions();

        size_t old_count = this->count;
        size_t copy_count = std::min(old_count, new_count);

        // Save active data locally before releasing all MR resources.
        // This lets us deregister every MR slot before fi_mr_reg, avoiding
        // CXI key exhaustion (ENOKEY).
        std::vector<unsigned char> saved;
        if(copy_count > 0)
        {
            saved.resize(copy_count * sizeof(T));
            std::memcpy(saved.data(), this->buffer, saved.size());
        }

        // Reset staging pointer but keep the MR registered to avoid
        // consuming a new provider key slot on the next use.
        this->ResetStaging();

        for(RetiredBuffer &retired : this->retired_buffers)
        {
            if(retired.mr)
            {
                this->context.DeregisterMemory(retired.mr);
            }
            if(retired.buffer)
            {
                rma_detail::advise_dontneed(retired.buffer, retired.count * sizeof(T));
                std::free(retired.buffer);
            }
        }
        this->retired_buffers.clear();

        if(this->mr)
        {
            this->context.DeregisterMemory(this->mr);
            this->mr = nullptr;
        }
        if(this->buffer)
        {
            rma_detail::advise_dontneed(this->buffer, old_count * sizeof(T));
            std::free(this->buffer);
            this->buffer = nullptr;
        }

        // Allocate and register new buffer with freed MR slots
        size_t new_alloc_size = new_count * sizeof(T);
        if(new_alloc_size == 0) new_alloc_size = sizeof(T);
        size_t new_aligned_size = ((new_alloc_size + 63) / 64) * 64;

        T *new_buffer = static_cast<T*>(std::aligned_alloc(64, new_aligned_size));
        if(not new_buffer)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent::LocalResize: aligned_alloc failed");
        }
        std::memset(new_buffer, 0, new_aligned_size);

        fid_mr *new_mr = this->context.RegisterMemory(new_buffer, new_aligned_size,
            FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
        if(not new_mr)
        {
            std::free(new_buffer);
            throw std::runtime_error("OFIRemoteMemoryAgent::LocalResize: fi_mr_reg failed");
        }

        if(copy_count > 0)
        {
            std::memcpy(new_buffer, saved.data(), saved.size());
        }

        this->buffer = new_buffer;
        this->mr = new_mr;
        this->count = new_count;

        if(this->my_agent_rank >= 0 and this->my_agent_rank < static_cast<int>(this->remote_regions.size()))
        {
            this->remote_regions[this->my_agent_rank].addr = this->context.UsesVirtAddr()
                ? reinterpret_cast<uint64_t>(this->buffer) : 0;
            this->remote_regions[this->my_agent_rank].key = fi_mr_key(this->mr);
        }

        return this->GetLocalRemoteInfo();
    }

    RemoteBufferInfo GetLocalRemoteInfo() const override
    {
        RemoteBufferInfo info;
        info.addr = this->context.UsesVirtAddr() ? reinterpret_cast<uint64_t>(this->buffer) : 0;
        info.rkey = fi_mr_key(this->mr);
        info.count = this->count;
        return info;
    }

    void UpdateRemoteInfo(int peer_rank, const RemoteBufferInfo &info) override
    {
        if(peer_rank < 0 or peer_rank >= static_cast<int>(this->remote_regions.size()))
        {
            throw std::runtime_error("OFIRemoteMemoryAgent::UpdateRemoteInfo: peer rank is out of range");
        }
        this->remote_regions[peer_rank].addr = info.addr;
        this->remote_regions[peer_rank].key = info.rkey;
    }

    void Replace(size_t new_count) override
    {
        if(not this->owns_memory)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent::Replace: cannot replace user-supplied memory");
        }
        this->context.DrainCompletions();

        this->FreeStaging();

        if(this->mr)
        {
            this->context.DeregisterMemory(this->mr);
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
            throw std::runtime_error("OFIRemoteMemoryAgent::Replace: aligned_alloc failed");
        }
        std::memset(this->buffer, 0, new_aligned_size);

        this->mr = this->context.RegisterMemory(this->buffer, new_aligned_size,
            FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
        if(not this->mr)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent::Replace: fi_mr_reg failed");
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

        this->FreeStaging();

        if(this->scratch_mr)
        {
            this->context.DeregisterMemory(this->scratch_mr);
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
            this->context.DeregisterMemory(this->mr);
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

        for(auto &[key, reg] : this->external_registrations)
        {
            this->context.DeregisterMemory(reg.mr);
        }
        this->external_registrations.clear();

        for(RetiredBuffer &retired : this->retired_buffers)
        {
            if(retired.mr)
            {
                this->context.DeregisterMemory(retired.mr);
            }
            if(retired.buffer)
            {
                rma_detail::advise_dontneed(retired.buffer, retired.count * sizeof(T));
                std::free(retired.buffer);
            }
        }
        this->retired_buffers.clear();

        this->count = 0;
        this->staging_size = 0;
        this->staging_next = 0;
        this->staging_active_target = -1;
        this->freed = true;
    }

    static std::unique_ptr<OFIRemoteMemoryAgent<T>> Create(size_t count, OFIContext &context, MPI_Comm agent_comm)
    {
        return std::make_unique<OFIRemoteMemoryAgent<T>>(count, context, agent_comm);
    }

private:
    size_t count;
    OFIContext &context;
    MPI_Comm agent_comm;
    int my_agent_rank;
    std::vector<int> rank_map;
    T *buffer;
    fid_mr *mr;
    uint64_t *scratch;
    fid_mr *scratch_mr;
    mutable T *staging;
    mutable fid_mr *staging_mr;
    mutable size_t staging_size;
    mutable size_t staging_next;
    mutable int staging_active_target;
    std::vector<OFIRemoteRegion> remote_regions;

    struct RegisteredMR
    {
        fid_mr *mr;
        void *desc;
    };
    std::unordered_map<uint32_t, RegisteredMR> external_registrations;
    uint32_t next_ext_key;

    struct RetiredBuffer
    {
        T *buffer;
        fid_mr *mr;
        size_t count;
    };
    std::vector<RetiredBuffer> retired_buffers;
    bool freed;
    bool owns_memory;

    static constexpr size_t DIRECT_REG_BYTE_THRESHOLD = 8192;

    void *BufferDesc() const { return fi_mr_desc(this->mr); }
    void *ScratchDesc() const { return fi_mr_desc(this->scratch_mr); }
    void *StagingDesc() const { return fi_mr_desc(this->staging_mr); }
    uint64_t BufferKey() const { return fi_mr_key(this->mr); }

    void ResetStaging() const
    {
        this->staging_next = 0;
        this->staging_active_target = -1;
    }

    void FreeStaging()
    {
        if(this->staging_mr)
        {
            this->context.DeregisterMemory(this->staging_mr);
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
            this->context.DeregisterMemory(this->staging_mr);
            this->staging_mr = nullptr;
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
            throw std::runtime_error("OFIRemoteMemoryAgent: aligned_alloc failed for staging");
        }
        std::memset(this->staging, 0, aligned_bytes);

        this->staging_mr = this->context.RegisterMemory(this->staging, aligned_bytes,
            FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
        if(not this->staging_mr)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent: fi_mr_reg failed for staging");
        }
        this->staging_size = new_size;
        this->staging_next = 0;
    }

    void RegisterUserBuffer(size_t count)
    {
        size_t reg_size = count * sizeof(T);
        if(reg_size == 0) reg_size = sizeof(T);

        this->mr = this->context.RegisterMemory(this->buffer, reg_size,
            FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
        if(not this->mr)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent: fi_mr_reg failed for user buffer");
        }

        this->scratch = static_cast<uint64_t*>(std::aligned_alloc(64, 64));
        if(not this->scratch)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent: aligned_alloc failed for scratch");
        }

        this->scratch_mr = this->context.RegisterMemory(this->scratch, 64,
            FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
        if(not this->scratch_mr)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent: fi_mr_reg failed for scratch");
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
            throw std::runtime_error("OFIRemoteMemoryAgent: aligned_alloc failed for buffer");
        }
        std::memset(this->buffer, 0, aligned_size);

        this->mr = this->context.RegisterMemory(this->buffer, aligned_size,
            FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
        if(not this->mr)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent: fi_mr_reg failed for buffer");
        }

        this->scratch = static_cast<uint64_t*>(std::aligned_alloc(64, 64));
        if(not this->scratch)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent: aligned_alloc failed for scratch");
        }

        this->scratch_mr = this->context.RegisterMemory(this->scratch, 64,
            FI_REMOTE_READ | FI_REMOTE_WRITE | FI_READ | FI_WRITE);
        if(not this->scratch_mr)
        {
            throw std::runtime_error("OFIRemoteMemoryAgent: fi_mr_reg failed for scratch");
        }
    }

    void ExchangeRemoteInfo()
    {
        OFIRemoteRegion local_info{};
        local_info.addr = this->context.UsesVirtAddr() ? reinterpret_cast<uint64_t>(this->buffer) : 0;
        local_info.key = this->BufferKey();

        int size;
        MPI_Comm_size(this->agent_comm, &size);
        this->remote_regions.resize(size);

        MPI_Allgather(&local_info, sizeof(OFIRemoteRegion), MPI_BYTE,
                      this->remote_regions.data(), sizeof(OFIRemoteRegion), MPI_BYTE,
                      this->agent_comm);
    }

};

#endif // __WITH_OFI

#endif // OFI_REMOTE_MEMORY_AGENT_HPP
