#ifndef GLOBAL_COUNTER_HPP
#define GLOBAL_COUNTER_HPP

#ifdef __WITH_MPI

#include <mpi_utils/mpi_commands.hpp>

class GlobalCounter
{
public:
    GlobalCounter(const MPI_Comm &comm, int globalInitialValue);

    ~GlobalCounter();

    void Set(int n);

    void Destroy(void);

    int Increment(int n);

    inline int Decrement(int n = 1){return this->Increment(-n);};

    inline int GetValue(void) const{return *this->counter;};

private:
    MPI_Comm comm;
    rank_t rank, size, master_rank;
    volatile int *counter;
    MPI_Win counter_win;
    bool destroyed;
};

#endif // __WITH_MPI

#endif // GLOBAL_COUNTER_HPP