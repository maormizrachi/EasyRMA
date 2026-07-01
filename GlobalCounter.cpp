#ifdef __WITH_MPI

#include "GlobalCounter.hpp"

GlobalCounter::GlobalCounter(const MPI_Comm &comm, int globalInitialValue)
{
    this->comm = comm;
    MPI_Comm_rank(comm, &this->rank);
    MPI_Comm_size(comm, &this->size);

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

    this->master_rank = 0;
    bool master = (this->rank == this->master_rank);
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none"); // No strict ordering
    MPI_Info_set(info, "accumulate_ops", "same_op");
    MPI_Info_set(info, "same_disp_unit", "true");
    int err = MPI_Win_allocate((master)? sizeof(size_t) : 0, sizeof(size_t), info, comm, &this->counter, &this->counter_win);
    reportErrorAndExit("MPI_Win_allocate for GlobalCounter", err);
    
    MPI_Win_set_errhandler(this->counter_win, MPI_ERRORS_RETURN);
    MPI_Info_free(&info);

    int *model, flag;
    MPI_Win_get_attr(this->counter_win, MPI_WIN_MODEL, &model, &flag);
    if(*model == MPI_WIN_SEPARATE)
    {
        std::cout << "MPI is using WIN_SEPARATE (" << MPI_WIN_SEPARATE << "). Can not continue" << std::endl;
        exit(1);
    }

    if(this->rank == this->master_rank)
    {
        this->Set(globalInitialValue);
    }
    MPI_Barrier(comm);
}

void GlobalCounter::Destroy(void)
{
    if(this->destroyed)
    {
        return;
    }
    
    MPI_Win_free(&this->counter_win);
    MPI_Barrier(this->comm);
    this->destroyed = true;
}

GlobalCounter::~GlobalCounter()
{
    if(not std::uncaught_exceptions())
    {
        if(not this->destroyed)
        {
            this->Destroy();
        }
    }
}

void GlobalCounter::Set(int n)
{
    MPI_Win_lock(MPI_LOCK_SHARED, this->master_rank, MPI_MODE_NOCHECK, this->counter_win);
    MPI_Put(&n, 1, MPI_INT, this->master_rank, 0, 1, MPI_INT, this->counter_win);
    MPI_Win_unlock(this->master_rank, this->counter_win);
}

int GlobalCounter::Increment(int n)
{
    int result;
    MPI_Win_lock(MPI_LOCK_SHARED, this->master_rank, MPI_MODE_NOCHECK, this->counter_win);
    int retval = MPI_Fetch_and_op(&n, &result, MPI_INT, this->master_rank, 0, MPI_SUM, this->counter_win);
    assert(retval == MPI_SUCCESS);
    MPI_Win_unlock(this->master_rank, this->counter_win);
    return result;
}

#endif // __WITH_MPI