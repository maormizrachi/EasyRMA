#ifndef DISTRIBUTED_MUTEX_HPP
#define DISTRIBUTED_MUTEX_HPP

#include <mpi.h>
#include <memory>
#include <mpi_utils/mpi_commands.hpp>
#include "RMAFactory.hpp"

class DistributedMutex
{
public:
    DistributedMutex(const MPI_Comm &comm, rank_t rank, RDMA_Type rdma_type);
    
    ~DistributedMutex();

    void Lock(void);

    void Unlock(void);

    void Destroy(void);

private:
    MPI_Comm comm;
    rank_t rank;
    std::unique_ptr<RemoteMemoryAgent<int>> agent;
    bool destroyed;
};

#endif // DISTRIBUTED_MUTEX_HPP