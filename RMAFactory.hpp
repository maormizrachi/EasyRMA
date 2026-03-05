#ifndef RMA_FACTORY_HPP
#define RMA_FACTORY_HPP

#ifdef __WITH_MPI

#include "RemoteMemoryAgent.hpp"
#include "MPIRemoteMemoryAgent.hpp"
#ifdef __WITH_IBV
#include "IBVRemoteMemoryAgent.hpp"
#endif

#include <memory>
#include <stdexcept>

enum class RDMA_Type
{
    MPI_RMA,
    IBV_RDMA,
    AUTO_RDMA
};

class RMAFactory
{
public:
    RMAFactory() = delete;

    static RDMA_Type ResolveAutoRDMA()
    {
#ifdef OPEN_MPI
        return RDMA_Type::MPI_RMA;
#elif defined(__WITH_IBV)
        return RDMA_Type::IBV_RDMA;
#else
        return RDMA_Type::MPI_RMA;
#endif
    }

    template<typename T>
    static std::unique_ptr<RemoteMemoryAgent<T>> Create(RDMA_Type type, size_t count, MPI_Comm comm)
    {
        if(type == RDMA_Type::AUTO_RDMA)
        {
            type = ResolveAutoRDMA();
        }

        switch(type)
        {
            case RDMA_Type::MPI_RMA:
                return MPIRemoteMemoryAgent<T>::CreateWithDefaultInfo(count, comm);
            case RDMA_Type::IBV_RDMA:
#ifdef __WITH_IBV
                return CreateIBV<T>(count, comm);
#else
                throw std::runtime_error("RMAFactory: IBV_RDMA selected but __WITH_IBV is not enabled");
#endif
            default:
                break;
        }
        throw std::runtime_error("RMAFactory: unknown RDMA type");
    }

private:
#ifdef __WITH_IBV
    static std::shared_ptr<IBVContext> &GetSharedContext()
    {
        static std::shared_ptr<IBVContext> context;
        return context;
    }

    template<typename T>
    static std::unique_ptr<RemoteMemoryAgent<T>> CreateIBV(size_t count, MPI_Comm agent_comm)
    {
        auto &ctx = GetSharedContext();
        if(not ctx)
        {
            ctx = std::make_shared<IBVContext>(MPI_COMM_WORLD);
        }
        return IBVRemoteMemoryAgent<T>::Create(count, *ctx, agent_comm);
    }
#endif
};

#endif // __WITH_MPI

#endif // RMA_FACTORY_HPP
