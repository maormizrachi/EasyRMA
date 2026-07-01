#ifndef CONDITION_VARIABLE_HPP
#define CONDITION_VARIABLE_HPP

#ifdef __WITH_MPI

#include <stdexcept>
#include "DistributedMutex.hpp"

class ConditionVariable
{
public:
    ConditionVariable(const MPI_Comm &comm);
    
    ~ConditionVariable(void);

    void Wait(DistributedMutex &mutex, const std::function<void(void)> &work_function);
    
    void Notify(void);

    void Destroy(void);

private:
    MPI_Comm comm;
    rank_t internal_rank;
    rank_t other_rank;
    MPI_Win win;
    int *value;
    bool destroyed;
};

#endif // __WITH_MPI

#endif // CONDITION_VARIABLE_HPP