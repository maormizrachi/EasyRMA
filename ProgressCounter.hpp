#ifndef PROGRESS_COUNTER_HPP
#define PROGRESS_COUNTER_HPP

#ifdef __WITH_MPI

#include "GlobalCounter.hpp"

class ProgressCounter
{
public:
    ProgressCounter(const MPI_Comm &comm);

    ~ProgressCounter();

    void Destroy(void);

    void Reset(int myNumParticles);

    int Increment(int n);

    inline int Decrement(int n = 1){return this->Increment(-n);};
    
    void MarkDone(void);

    int GetValue(void) const{return this->counter->GetValue();};
    
    volatile int *is_done;
    int localDecrementAmount;
    
private:    
    std::shared_ptr<GlobalCounter> counter; // TODO: should be private
    rank_t rank, size, master_rank;
    MPI_Win is_done_win;
    MPI_Comm comm;
    bool destroyed;
};

#endif // __WITH_MPI

#endif // PROGRESS_COUNTER_HPP