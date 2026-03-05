#ifndef MPI_REMOTE_MEMORY_AGENT_HPP
#define MPI_REMOTE_MEMORY_AGENT_HPP

#ifdef __WITH_MPI

#include "RemoteMemoryAgent.hpp"
#include <mpi.h>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <memory>

namespace detail
{
    inline MPI_Datatype AtomicMPIType(size_t elem_size)
    {
        switch(elem_size)
        {
            case 4:  return MPI_INT;
            case 8:  return MPI_INT64_T;
            default:
                throw std::runtime_error("MPIRemoteMemoryAgent: atomic operations require 4- or 8-byte types, got " + std::to_string(elem_size) + " bytes");
        }
    }

    inline void CheckMPIError(int err, const char *context)
    {
        if(err == MPI_SUCCESS)
        {
            return;
        }
        char error_string[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(err, error_string, &length);
        throw std::runtime_error(std::string(context) + ": " + error_string);
    }

    inline MPI_Info CreateDefaultRMAInfo()
    {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "accumulate_ordering", "none");
        MPI_Info_set(info, "accumulate_ops", "same_op");
        MPI_Info_set(info, "same_disp_unit", "true");
        return info;
    }

    inline void ValidateUnifiedModel(MPI_Win win)
    {
        int *model = nullptr;
        int flag = 0;
        MPI_Win_get_attr(win, MPI_WIN_MODEL, &model, &flag);
        if(flag and model and *model == MPI_WIN_SEPARATE)
        {
            throw std::runtime_error("MPIRemoteMemoryAgent: MPI_WIN_SEPARATE memory model is not supported. MPI_WIN_UNIFIED is required for correct passive-target RMA.");
        }
    }
}

template<typename T>
class MPIRemoteMemoryAgent : public RemoteMemoryAgent<T>
{
public:
    MPIRemoteMemoryAgent(size_t count, MPI_Comm comm, MPI_Info info = MPI_INFO_NULL)
        : count(count), comm(comm), ptr(nullptr), win(MPI_WIN_NULL), freed(false)
    {
        this->AllocateWindow(count, info);
    }

    ~MPIRemoteMemoryAgent() override
    {
        if(not std::uncaught_exceptions() and not this->freed)
        {
            this->Free();
        }
    }

    T *GetLocalPointer() override
    {
        return this->ptr;
    }

    size_t GetCount() const override
    {
        return this->count;
    }

    void Put(const T *origin, size_t count, int target_rank, size_t target_disp, bool flush = true) override
    {
        size_t bytes = count * sizeof(T);
        size_t byte_offset = target_disp * sizeof(T);
        MPI_Put(origin, static_cast<int>(bytes), MPI_BYTE, target_rank, static_cast<MPI_Aint>(byte_offset), static_cast<int>(bytes), MPI_BYTE, this->win);
        if(flush)
        {
            MPI_Win_flush(target_rank, this->win);
        }
    }

    void Get(T *result, size_t count, int target_rank, size_t target_disp, bool flush = true) const override
    {
        size_t bytes = count * sizeof(T);
        size_t byte_offset = target_disp * sizeof(T);
        MPI_Get(result, static_cast<int>(bytes), MPI_BYTE, target_rank, static_cast<MPI_Aint>(byte_offset), static_cast<int>(bytes), MPI_BYTE, this->win);
        if(flush)
        {
            MPI_Win_flush(target_rank, this->win);
        }
    }

    void CompareAndSwap(const T &desired, const T &expected, T &old_value, int target_rank, size_t target_disp, bool flush = true) override
    {
        MPI_Datatype dt = detail::AtomicMPIType(sizeof(T));
        size_t byte_offset = target_disp * sizeof(T);
        MPI_Compare_and_swap(&desired, &expected, &old_value, dt, target_rank, static_cast<MPI_Aint>(byte_offset), this->win);
        if(flush)
        {
            MPI_Win_flush(target_rank, this->win);
        }
    }

    T FetchAndAdd(const T &addend, int target_rank,
                  size_t target_disp, bool flush = true) override
    {
        MPI_Datatype dt = detail::AtomicMPIType(sizeof(T));
        size_t byte_offset = target_disp * sizeof(T);
        T old_value;
        MPI_Fetch_and_op(&addend, &old_value, dt, target_rank, static_cast<MPI_Aint>(byte_offset), MPI_SUM, this->win);
        if(flush)
        {
            MPI_Win_flush(target_rank, this->win);
        }
        return old_value;
    }

    void Flush(int target_rank) override
    {
        MPI_Win_flush(target_rank, this->win);
    }

    void Resize(size_t new_count) override
    {
        MPI_Win new_win = MPI_WIN_NULL;
        T *new_ptr = nullptr;

        MPI_Info info = detail::CreateDefaultRMAInfo();

        int err = MPI_Win_allocate(static_cast<MPI_Aint>(new_count * sizeof(T)), 1, info, this->comm, &new_ptr, &new_win);
        detail::CheckMPIError(err, "MPIRemoteMemoryAgent::Resize MPI_Win_allocate");
        MPI_Info_free(&info);

        if(new_ptr == nullptr and new_count > 0)
        {
            throw std::runtime_error("MPIRemoteMemoryAgent::Resize: MPI_Win_allocate returned null");
        }

        detail::ValidateUnifiedModel(new_win);
        MPI_Win_set_errhandler(new_win, MPI_ERRORS_RETURN);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, new_win);

        size_t copy_count = std::min(this->count, new_count);
        if(copy_count > 0)
        {
            std::memcpy(new_ptr, this->ptr, copy_count * sizeof(T));
        }

        MPI_Win_unlock_all(this->win);
        MPI_Win_free(&this->win);

        this->win = new_win;
        this->ptr = new_ptr;
        this->count = new_count;
    }

    void Free() override
    {
        if(this->freed)
        {
            return;
        }
        MPI_Win_unlock_all(this->win);
        MPI_Win_free(&this->win);
        this->win = MPI_WIN_NULL;
        this->ptr = nullptr;
        this->count = 0;
        this->freed = true;
    }

    static std::unique_ptr<MPIRemoteMemoryAgent<T>> CreateWithDefaultInfo(size_t count, MPI_Comm comm)
    {
        MPI_Info info = detail::CreateDefaultRMAInfo();
        auto agent = std::make_unique<MPIRemoteMemoryAgent<T>>(count, comm, info);
        MPI_Info_free(&info);
        return agent;
    }

private:
    size_t count;
    MPI_Comm comm;
    T *ptr;
    MPI_Win win;
    bool freed;

    void AllocateWindow(size_t count, MPI_Info info)
    {
        bool using_default_info = (info == MPI_INFO_NULL);
        if(using_default_info)
        {
            info = detail::CreateDefaultRMAInfo();
        }

        int err = MPI_Win_allocate(static_cast<MPI_Aint>(count * sizeof(T)), 1, info, this->comm, &this->ptr, &this->win);
        detail::CheckMPIError(err, "MPIRemoteMemoryAgent constructor MPI_Win_allocate");

        if(using_default_info)
        {
            MPI_Info_free(&info);
        }

        if(this->ptr == nullptr and count > 0)
        {
            throw std::runtime_error("MPIRemoteMemoryAgent: MPI_Win_allocate returned null pointer");
        }

        detail::ValidateUnifiedModel(this->win);
        MPI_Win_set_errhandler(this->win, MPI_ERRORS_RETURN);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, this->win);
    }
};

#endif // __WITH_MPI

#endif // MPI_REMOTE_MEMORY_AGENT_HPP
