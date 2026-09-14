#ifdef __WITH_MPI

#include "DistributedMutex.hpp"
#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace
{
    const uint64_t NOT_INTERESTED = 0;
    const uint64_t INTERESTED = 1;
}

DistributedMutex::DistributedMutex(const MPI_Comm &comm, rank_t rank, RDMA_Type rdma_type):
    comm(comm), rank(rank), destroyed(false)
{
    assert(this->comm != MPI_COMM_NULL);
    rank_t size;
    MPI_Comm_rank(this->comm, &this->myRank);
    MPI_Comm_size(this->comm, &size);
    if(size != 2)
    {
        throw std::runtime_error("DistributedMutex: requires a two-rank communicator");
    }
    this->peerRank = 1 - this->myRank;

    this->agent = RMAFactory::Create<uint64_t>(rdma_type, DistributedMutex::SLOTS_NUM, this->comm);

    uint64_t *local = this->agent->GetLocalPointer();
    if(local != nullptr)
    {
        local[DistributedMutex::INTENT_SLOT] = NOT_INTERESTED;
        local[DistributedMutex::TURN_SLOT] = static_cast<uint64_t>(DistributedMutex::TURN_HOLDER);
        this->agent->SyncLocal();
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

void DistributedMutex::PublishIntent(uint64_t value)
{
    uint64_t *local = this->agent->GetLocalPointer();
    __atomic_store_n(&local[DistributedMutex::INTENT_SLOT], value, __ATOMIC_SEQ_CST);
    this->agent->SyncLocal();
}

void DistributedMutex::PublishTurn(rank_t turnRank)
{
    uint64_t value = static_cast<uint64_t>(turnRank);
    if(this->myRank == DistributedMutex::TURN_HOLDER)
    {
        uint64_t *local = this->agent->GetLocalPointer();
        __atomic_store_n(&local[DistributedMutex::TURN_SLOT], value, __ATOMIC_SEQ_CST);
        this->agent->SyncLocal();
    }
    else
    {
        // Flushed, so the turn is in place before the peer's intent is read.
        this->agent->Put(&value, 1, DistributedMutex::TURN_HOLDER, DistributedMutex::TURN_SLOT, true);
    }
}

void DistributedMutex::ReadPeerIntentAndTurn(uint64_t &peerIntent, rank_t &turnRank)
{
    if(this->myRank == DistributedMutex::TURN_HOLDER)
    {
        this->agent->Get(&peerIntent, 1, this->peerRank, DistributedMutex::INTENT_SLOT);
        uint64_t *local = this->agent->GetLocalPointer();
        turnRank = static_cast<rank_t>(__atomic_load_n(&local[DistributedMutex::TURN_SLOT], __ATOMIC_SEQ_CST));
    }
    else
    {
        // The peer's intent word and the turn word are adjacent in the turn
        // holder's region, so a single round trip reads both.
        uint64_t values[DistributedMutex::SLOTS_NUM];
        this->agent->Get(values, DistributedMutex::SLOTS_NUM, this->peerRank, 0);
        peerIntent = values[DistributedMutex::INTENT_SLOT];
        turnRank = static_cast<rank_t>(values[DistributedMutex::TURN_SLOT]);
    }
}

void DistributedMutex::Lock(void)
{
    int probe_flag;

    this->PublishIntent(INTERESTED);
    // Yield the turn, so two simultaneous acquires resolve to a single winner.
    this->PublishTurn(this->peerRank);

    while(true)
    {
        uint64_t peerIntent = NOT_INTERESTED;
        rank_t turnRank = this->myRank;
        this->ReadPeerIntentAndTurn(peerIntent, turnRank);
        if(peerIntent == NOT_INTERESTED or turnRank != this->peerRank)
        {
            break;
        }
        this->agent->MakeProgress();
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, this->comm, &probe_flag, MPI_STATUS_IGNORE);
    }
}

void DistributedMutex::Unlock(void)
{
    this->PublishIntent(NOT_INTERESTED);
}

void DistributedMutex::MakeProgress(void)
{
    if(this->agent)
    {
        this->agent->MakeProgress();
    }
}

#endif // __WITH_MPI
