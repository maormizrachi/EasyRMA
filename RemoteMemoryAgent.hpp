#ifndef REMOTE_MEMORY_AGENT_HPP
#define REMOTE_MEMORY_AGENT_HPP

#include <cstddef>
#include <cstdint>

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

    virtual void Resize(size_t new_count) = 0;

    //! Free the current buffer and allocate a fresh one of size new_count.
    //! Unlike Resize, this does NOT copy old data — use only when contents are not needed.
    //! This avoids peak memory = old + new that Resize incurs.
    virtual void Replace(size_t new_count) = 0;

    virtual void Free() = 0;
};

#endif // REMOTE_MEMORY_AGENT_HPP
