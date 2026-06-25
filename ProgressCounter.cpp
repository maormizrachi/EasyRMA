#include "ProgressCounter.hpp"

ProgressCounter::ProgressCounter(const MPI_Comm &comm)
    : comm(comm), destroyed(false)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);

    this->master_rank = 0;

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none"); // No strict ordering
    MPI_Info_set(info, "accumulate_ops", "same_op");
    MPI_Info_set(info, "same_disp_unit", "true");
    MPI_Win_allocate(sizeof(int), sizeof(int), info, comm, &this->is_done, &this->is_done_win);
    MPI_Win_set_errhandler(this->is_done_win, MPI_ERRORS_RETURN);
    MPI_Info_free(&info);

    int *model, flag;
    MPI_Win_get_attr(this->is_done_win, MPI_WIN_MODEL, &model, &flag);
    if(*model == MPI_WIN_SEPARATE)
    {
        std::cout << "MPI is using WIN_SEPARATE (" << MPI_WIN_SEPARATE << "). Can not continue" << std::endl;
        exit(1);
    }
    
    this->counter = std::make_shared<GlobalCounter>(comm, 0);
    
    MPI_Barrier(comm);
}

void ProgressCounter::Reset(int myNumParticles)
{
    int totalNumParticles = 0;
    MPI_Reduce(&myNumParticles, (void*)&totalNumParticles, 1, MPI_INT, MPI_SUM, this->master_rank, this->comm);

    if(this->rank == this->master_rank)
    {
        this->counter->Set(totalNumParticles);
    }

    // reset `is_done`
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->is_done_win);
    int zero = 0;
    MPI_Put(&zero, 1, MPI_INT, this->rank, 0, 1, MPI_INT, this->is_done_win);
    MPI_Win_unlock(this->rank, this->is_done_win);

    MPI_Barrier(this->comm);
}

void ProgressCounter::Destroy(void)
{
    if(this->destroyed)
    {
        return;
    }
    
    this->counter->Destroy();
    MPI_Win_free(&this->is_done_win);
    this->destroyed = true;
}

ProgressCounter::~ProgressCounter()
{
    if(not std::uncaught_exceptions())
    {
        if(not this->destroyed)
        {
            this->Destroy();
        }
    }
}

int ProgressCounter::Increment(int n)
{
    int result = this->counter->Increment(n);
    int currValue = result + n;
    if(currValue == 0)
    {
        this->MarkDone();
    }
    return currValue;
}

void ProgressCounter::MarkDone(void)
{
    static int plus_one = 1;
    MPI_Win_lock_all(MPI_MODE_NOCHECK, this->is_done_win);
    for(rank_t _rank = 0; _rank < this->size; _rank++)
    {
        // MPI_Win_lock(MPI_LOCK_SHARED, _rank, MPI_MODE_NOCHECK, this->is_done_win);
        MPI_Put(&plus_one, 1, MPI_INT, _rank, 0, 1, MPI_INT, this->is_done_win);
        // MPI_Win_unlock(_rank, this->is_done_win);
    }
    MPI_Win_flush_all(this->is_done_win);
    MPI_Win_unlock_all(this->is_done_win);
}