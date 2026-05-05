#ifndef REMOTE_MEMORY_AGENT_HPP
#define REMOTE_MEMORY_AGENT_HPP

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <sys/mman.h>

namespace rma_detail {
    inline void advise_dontneed(void *ptr, size_t bytes)
    {
        if(not ptr or bytes == 0) return;

        constexpr size_t page_size = 4096;
        uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t end   = start + bytes;

        uintptr_t page_start = (start + page_size - 1) & ~(page_size - 1);
        uintptr_t page_end   = end & ~(page_size - 1);

        if(page_start < page_end)
        {
            madvise(reinterpret_cast<void*>(page_start),
                    page_end - page_start, MADV_DONTNEED);
        }
    }
} // namespace rma_detail

template<typename T>
class RemoteMemoryAgent
{
public:
    virtual ~RemoteMemoryAgent() = default;

    virtual T *GetLocalPointer() = 0;

    virtual size_t GetCount() const = 0;

    virtual void Put(const T *origin, size_t count, int target_rank, size_t target_disp, bool flush = true) = 0;

    virtual void PutScatter(const T *contiguous_source, const uint32_t *target_disps, size_t count, int target_rank, bool flush = true)
    {
        for(size_t i = 0; i < count; i++)
        {
            this->Put(&contiguous_source[i], 1, target_rank,
                      static_cast<size_t>(target_disps[i]), false);
        }
        if(flush)
        {
            this->Flush(target_rank);
        }
    }

    virtual void Get(T *result, size_t count, int target_rank, size_t target_disp, bool flush = true) const = 0;

    virtual void CompareAndSwap(const T &desired, const T &expected, T &old_value, int target_rank, size_t target_disp, bool flush = true) = 0;

    virtual T FetchAndAdd(const T &addend, int target_rank, size_t target_disp, bool flush = true) = 0;

    virtual void Flush(int target_rank) = 0;

    virtual void SyncLocal()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    virtual void Resize(size_t new_count) = 0;

    // Free the current buffer and allocate a fresh one of size new_count.
    // Unlike Resize, this does NOT copy old data — use only when contents are not needed.
    // This avoids peak memory = old + new that Resize incurs.
    virtual void Replace(size_t new_count) = 0;

    virtual void Free() = 0;
};

#endif // REMOTE_MEMORY_AGENT_HPP
