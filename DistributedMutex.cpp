#ifdef __WITH_MPI

#include "DistributedMutex.hpp"
#include <cassert>
#include <cstdint>

DistributedMutex::DistributedMutex(const MPI_Comm &comm, rank_t rank, RDMA_Type rdma_type):
    comm(comm), rank(rank), destroyed(false)
{
    assert(this->comm != MPI_COMM_NULL);
    rank_t my_rank, size;
    MPI_Comm_rank(this->comm, &my_rank);
    MPI_Comm_size(this->comm, &size);
    assert(size > 1);

    this->agent = RMAFactory::Create<uint64_t>(rdma_type, 1, this->comm);

    if(my_rank == rank and this->agent->GetLocalPointer() != nullptr)
    {
        *this->agent->GetLocalPointer() = 0;
    }

    MPI_Barrier(this->comm);
}

void DistributedMutex::Destroy()
{
    if(this->destroyed)
    {
        return;
    }
    this->agent->Free();
    this->destroyed = true;
}

DistributedMutex::~DistributedMutex()
{
    if(not std::uncaught_exceptions())
    {
        if(not this->destroyed)
        {
            this->Destroy();
        }
    }
}

void DistributedMutex::Lock(void)
{
    const uint64_t one = 1;
    const uint64_t zero = 0;
    uint64_t old = 0;
    int probe_flag;

    while(true)
    {
        this->agent->CompareAndSwap(one, zero, old, this->rank, 0);
        if(old == 0)
        {
            break;
        }
        this->agent->MakeProgress();
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, this->comm, &probe_flag, MPI_STATUS_IGNORE);
    }
}

void DistributedMutex::Unlock(void)
{
    const uint64_t zero = 0;
    const uint64_t one = 1;
    uint64_t old;
    this->agent->CompareAndSwap(zero, one, old, this->rank, 0);
}

void DistributedMutex::MakeProgress(void)
{
    if(this->agent)
    {
        this->agent->MakeProgress();
    }
}

#endif // __WITH_MPI
