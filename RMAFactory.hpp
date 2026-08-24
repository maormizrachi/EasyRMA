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
#include <exception>
#include <cstdio>
#include <utility>

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
#elif defined(__WITH_IBV)
        return RDMA_Type::IBV_RDMA;
#else
        return RDMA_Type::MPI_RMA;
#endif
    }

    // Native verbs memory regions cannot grow in place: replacing one
    // invalidates its rkey immediately. Dynamic queue payloads therefore use
    // MPI RMA on IBV systems, whose collective window replacement supplies
    // the required lifetime synchronization. Fixed-size control regions still
    // use Create/CreateOver and remain native IBV. OFI (including CXI) is
    // unchanged.
    static RDMA_Type ResolveResizableRDMA(RDMA_Type type)
    {
        if(type == RDMA_Type::AUTO_RDMA)
        {
            type = ResolveAutoRDMA();
        }
        return type == RDMA_Type::IBV_RDMA ? RDMA_Type::MPI_RMA : type;
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
                local_available = OFIContext::HasUsableProvider("", comm) ? 1 : 0;
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

    static void Initialize(RDMA_Type type, MPI_Comm comm)
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
                int local_initialized = ctx ? 1 : 0;
                int min_initialized = 0;
                int max_initialized = 0;
                MPI_Allreduce(&local_initialized, &min_initialized, 1, MPI_INT, MPI_MIN, comm);
                MPI_Allreduce(&local_initialized, &max_initialized, 1, MPI_INT, MPI_MAX, comm);
                if(min_initialized != max_initialized)
                {
                    throw std::runtime_error("RMAFactory::Initialize: OFI context is initialized on only some ranks");
                }
                if(ctx)
                {
                    int relation = MPI_UNEQUAL;
                    MPI_Comm_compare(ctx->GetComm(), comm, &relation);
                    int local_compatible =
                        (relation == MPI_IDENT or relation == MPI_CONGRUENT) ? 1 : 0;
                    int all_compatible = 0;
                    MPI_Allreduce(&local_compatible, &all_compatible, 1, MPI_INT, MPI_MIN, comm);
                    if(not all_compatible)
                    {
                        throw std::runtime_error(
                            "RMAFactory::Initialize: existing OFI context uses a different communicator");
                    }
                }
                else
                {
                    std::shared_ptr<OFIContext> new_ctx;
                    std::exception_ptr init_error;
                    try
                    {
                        new_ctx = std::make_shared<OFIContext>(comm);
                    }
                    catch(...)
                    {
                        init_error = std::current_exception();
                    }

                    int local_success = init_error ? 0 : 1;
                    int all_success = 0;
                    MPI_Allreduce(&local_success, &all_success, 1, MPI_INT, MPI_MIN, comm);
                    if(not all_success)
                    {
                        new_ctx.reset();
                        if(init_error)
                        {
                            try
                            {
                                std::rethrow_exception(init_error);
                            }
                            catch(const std::exception &e)
                            {
                                int rank = 0;
                                MPI_Comm_rank(comm, &rank);
                                fprintf(stderr, "[OFI rank %d] context initialization failed: %s\n",
                                        rank, e.what());
                            }
                            catch(...)
                            {
                            }
                        }
                        throw std::runtime_error(
                            "RMAFactory::Initialize: OFI context initialization failed on at least one rank");
                    }
                    ctx = std::move(new_ctx);
                }
                break;
            }
#else
                throw std::runtime_error("RMAFactory: OFI_RDMA selected but __WITH_OFI is not enabled");
#endif
            default:
                break;
        }
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

    // Destroy shared provider contexts while MPI and accelerator runtimes are
    // still alive. Leaving these function-local statics to process teardown
    // makes libfabric free CXI resources after MPI_Finalize.
    static void Finalize(RDMA_Type type)
    {
        if(type == RDMA_Type::AUTO_RDMA)
        {
            type = ResolveAutoRDMA();
        }

        switch(type)
        {
            case RDMA_Type::OFI_RDMA:
#ifdef __WITH_OFI
                GetSharedOFIContext().reset();
#endif
                break;
            case RDMA_Type::IBV_RDMA:
#ifdef __WITH_IBV
                GetSharedIBVContext().reset();
#endif
                break;
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
    static std::unique_ptr<RemoteMemoryAgent<T>> CreateResizable(RDMA_Type type, size_t count, MPI_Comm comm)
    {
        return Create<T>(ResolveResizableRDMA(type), count, comm);
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
            throw std::runtime_error(
                "RMAFactory::CreateOFI: call RMAFactory::Initialize(OFI_RDMA, comm) "
                "collectively before creating OFI agents");
        }
        return OFIRemoteMemoryAgent<T>::Create(count, *ctx, agent_comm);
    }

    template<typename T>
    static std::unique_ptr<RemoteMemoryAgent<T>> CreateOFIOver(T *user_buffer, size_t count, MPI_Comm agent_comm)
    {
        auto &ctx = GetSharedOFIContext();
        if(not ctx)
        {
            throw std::runtime_error(
                "RMAFactory::CreateOFIOver: call RMAFactory::Initialize(OFI_RDMA, comm) "
                "collectively before creating OFI agents");
        }
        return std::make_unique<OFIRemoteMemoryAgent<T>>(user_buffer, count, *ctx, agent_comm);
    }
#endif
};

#endif // __WITH_MPI

#endif // RMA_FACTORY_HPP
