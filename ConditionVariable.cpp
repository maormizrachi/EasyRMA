#ifdef __WITH_MPI

#include "ConditionVariable.hpp"

ConditionVariable::ConditionVariable(const MPI_Comm &comm)
    : comm(comm), destroyed(false)
{
    MPI_Comm_rank(comm, &this->internal_rank);
    rank_t size;
    MPI_Comm_size(comm, &size);
    if(size != 2)
    {
        throw std::runtime_error("ConditionVariable only works with 2 ranks");
    }

    auto reportErrorAndExit = [](const std::string &str, int err)
    {
        if(err == MPI_SUCCESS)
        {
            return;
        }
        char error_string[MPI_MAX_ERROR_STRING];
        int length_of_error_string;
        MPI_Error_string(err, error_string, &length_of_error_string);
        std::cerr << "Error: " << str << ": " << error_string << std::endl;
        exit(1);
    };

    this->other_rank = 1 - this->internal_rank;
    int err = MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm, &this->value, &this->win);
    reportErrorAndExit("MPI_Win_allocate", err);
    *this->value = 0;
    MPI_Barrier(this->comm);
}

ConditionVariable::~ConditionVariable(void)
{
    if(not std::uncaught_exceptions())
    {
        if(not this->destroyed)
        {
            this->Destroy();
        }
    }
}

void ConditionVariable::Destroy(void)
{
    if(this->destroyed)
    {
        return;
    }
    // std::cout << "Destroys cond var" << std::endl;
    MPI_Barrier(this->comm);
    MPI_Win_free(&this->win);
    this->destroyed = true;
}

void ConditionVariable::Wait(DistributedMutex &mutex, const std::function<void(void)> &work_function)
{
    mutex.Unlock();
    int &value = *this->value;
    while(value == 0)
    {
        work_function();
        MPI_Win_lock(MPI_LOCK_SHARED, this->internal_rank, MPI_MODE_NOCHECK, this->win);
        MPI_Win_sync(this->win);
        MPI_Win_unlock(this->internal_rank, this->win);
    }
    // out! reset value
    value = 0;
    MPI_Barrier(this->comm);
    mutex.Lock();
}

void ConditionVariable::Notify(void)
{
    static int one = 1;
    MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->win);
    MPI_Put(&one, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, this->win);
    MPI_Win_unlock(this->other_rank, this->win);
    MPI_Barrier(this->comm);
}

#endif // __WITH_MPI