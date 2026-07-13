#ifndef RMA_FACTORY_HPP
#define RMA_FACTORY_HPP

#ifdef __WITH_MPI

#include "RemoteMemoryAgent.hpp"
#include "MPIRemoteMemoryAgent.hpp"
#ifdef __WITH_IBV
#include "IBVRemoteMemoryAgent.hpp"
#endif
#ifdef __WITH_OFI
#include "OFIRemoteMemoryAgent.hpp"
#endif

#include <memory>
#include <stdexcept>

enum class RDMA_Type
{
    MPI_RMA,
    IBV_RDMA,
    OFI_RDMA,
    AUTO_RDMA
};

class RMAFactory
{
public:
    RMAFactory() = delete;

    static RDMA_Type ResolveAutoRDMA()
    {
#ifdef __WITH_OFI
        return RDMA_Type::OFI_RDMA;
#else
        return RDMA_Type::MPI_RMA;
#endif
    }

    static bool IsBackendAvailable(RDMA_Type type, MPI_Comm comm)
    {
        if(type == RDMA_Type::AUTO_RDMA)
        {
            type = ResolveAutoRDMA();
        }

        int local_available = 0;
        switch(type)
        {
            case RDMA_Type::MPI_RMA:
                local_available = 1;
                break;
            case RDMA_Type::IBV_RDMA:
#ifdef __WITH_IBV
                local_available = 1;
#else
                local_available = 0;
#endif
                break;
            case RDMA_Type::OFI_RDMA:
#ifdef __WITH_OFI
                local_available = OFIContext::HasUsableProvider() ? 1 : 0;
#else
                local_available = 0;
#endif
                break;
            default:
                local_available = 0;
                break;
        }

        int global_available = 0;
        MPI_Allreduce(&local_available, &global_available, 1, MPI_INT, MPI_MIN, comm);
        return global_available != 0;
    }

    static void MakeProgress(RDMA_Type type)
    {
        if(type == RDMA_Type::AUTO_RDMA)
        {
            type = ResolveAutoRDMA();
        }

        switch(type)
        {
            case RDMA_Type::OFI_RDMA:
#ifdef __WITH_OFI
            {
                auto &ctx = GetSharedOFIContext();
                if(ctx)
                {
                    ctx->MakeProgress();
                }
                break;
            }
#else
                break;
#endif
            default:
                break;
        }
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
            case RDMA_Type::OFI_RDMA:
#ifdef __WITH_OFI
                return CreateOFI<T>(count, comm);
#else
                throw std::runtime_error("RMAFactory: OFI_RDMA selected but __WITH_OFI is not enabled");
#endif
            default:
                break;
        }
        throw std::runtime_error("RMAFactory: unknown RDMA type");
    }

    template<typename T>
    static std::unique_ptr<RemoteMemoryAgent<T>> CreateOver(RDMA_Type type, T *user_buffer, size_t count, MPI_Comm comm)
    {
        if(type == RDMA_Type::AUTO_RDMA)
        {
            type = ResolveAutoRDMA();
        }

        switch(type)
        {
            case RDMA_Type::MPI_RMA:
            {
                MPI_Info info = detail::CreateDefaultRMAInfo();
                auto agent = std::make_unique<MPIRemoteMemoryAgent<T>>(user_buffer, count, comm, info);
                MPI_Info_free(&info);
                return agent;
            }
            case RDMA_Type::IBV_RDMA:
#ifdef __WITH_IBV
                return CreateIBVOver<T>(user_buffer, count, comm);
#else
                throw std::runtime_error("RMAFactory: IBV_RDMA selected but __WITH_IBV is not enabled");
#endif
            case RDMA_Type::OFI_RDMA:
#ifdef __WITH_OFI
                return CreateOFIOver<T>(user_buffer, count, comm);
#else
                throw std::runtime_error("RMAFactory: OFI_RDMA selected but __WITH_OFI is not enabled");
#endif
            default:
                break;
        }
        throw std::runtime_error("RMAFactory: unknown RDMA type");
    }

private:
#ifdef __WITH_IBV
    static std::shared_ptr<IBVContext> &GetSharedIBVContext()
    {
        static std::shared_ptr<IBVContext> context;
        return context;
    }

    template<typename T>
    static std::unique_ptr<RemoteMemoryAgent<T>> CreateIBV(size_t count, MPI_Comm agent_comm)
    {
        auto &ctx = GetSharedIBVContext();
        if(not ctx)
        {
            ctx = std::make_shared<IBVContext>(MPI_COMM_WORLD);
        }
        return IBVRemoteMemoryAgent<T>::Create(count, *ctx, agent_comm);
    }

    template<typename T>
    static std::unique_ptr<RemoteMemoryAgent<T>> CreateIBVOver(T *user_buffer, size_t count, MPI_Comm agent_comm)
    {
        auto &ctx = GetSharedIBVContext();
        if(not ctx)
        {
            ctx = std::make_shared<IBVContext>(MPI_COMM_WORLD);
        }
        return std::make_unique<IBVRemoteMemoryAgent<T>>(user_buffer, count, *ctx, agent_comm);
    }
#endif

#ifdef __WITH_OFI
    static std::shared_ptr<OFIContext> &GetSharedOFIContext()
    {
        static std::shared_ptr<OFIContext> context;
        return context;
    }

    template<typename T>
    static std::unique_ptr<RemoteMemoryAgent<T>> CreateOFI(size_t count, MPI_Comm agent_comm)
    {
        auto &ctx = GetSharedOFIContext();
        if(not ctx)
        {
            ctx = std::make_shared<OFIContext>(MPI_COMM_WORLD);
        }
        return OFIRemoteMemoryAgent<T>::Create(count, *ctx, agent_comm);
    }

    template<typename T>
    static std::unique_ptr<RemoteMemoryAgent<T>> CreateOFIOver(T *user_buffer, size_t count, MPI_Comm agent_comm)
    {
        auto &ctx = GetSharedOFIContext();
        if(not ctx)
        {
            ctx = std::make_shared<OFIContext>(MPI_COMM_WORLD);
        }
        return std::make_unique<OFIRemoteMemoryAgent<T>>(user_buffer, count, *ctx, agent_comm);
    }
#endif
};

#endif // __WITH_MPI

#endif // RMA_FACTORY_HPP
