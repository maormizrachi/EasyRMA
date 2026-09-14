#ifndef DISTRIBUTED_MUTEX_HPP
#define DISTRIBUTED_MUTEX_HPP

#ifdef __WITH_MPI

#include <cstdint>
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

    void MakeProgress(void);

    void Destroy(void);

private:
    static const size_t INTENT_SLOT = 0;
    static const size_t TURN_SLOT = 1;
    static const size_t SLOTS_NUM = 2;
    static const rank_t TURN_HOLDER = 0;

    void PublishIntent(uint64_t value);

    void PublishTurn(rank_t turnRank);

    void ReadPeerIntentAndTurn(uint64_t &peerIntent, rank_t &turnRank);

    MPI_Comm comm;
    rank_t rank;
    rank_t myRank;
    rank_t peerRank;
    std::unique_ptr<RemoteMemoryAgent<uint64_t>> agent;
    bool destroyed;
};

#endif // __WITH_MPI

#endif // DISTRIBUTED_MUTEX_HPP
