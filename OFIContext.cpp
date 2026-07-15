#include "OFIContext.hpp"

#ifdef __WITH_OFI

#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <exception>
#include <climits>
#include <sys/uio.h>

#if defined(__WITH_PALS) && __has_include(<pals.h>) && __has_include(<rdma/fi_cxi_ext.h>)
#define STORM_WITH_CXI_PALS_AUTH 1
extern "C" {
#include <pals.h>
}
#include <rdma/fi_cxi_ext.h>
#endif

static void ThrowOFIError(const std::string &context, int err = 0)
{
    std::string msg = "OFIContext: " + context;
    if(err != 0)
    {
        msg += " (" + std::string(fi_strerror(err)) + ")";
    }
    throw std::runtime_error(msg);
}

static ssize_t PostWriteWithCompletion(fid_ep *ep, fi_addr_t peer_addr,
                                       const void *local_addr, size_t bytes,
                                       void *desc, uint64_t remote_addr,
                                       uint64_t rkey)
{
    iovec iov{};
    iov.iov_base = const_cast<void*>(local_addr);
    iov.iov_len = bytes;

    void *descs[1] = {desc};

    fi_rma_iov rma_iov{};
    rma_iov.addr = remote_addr;
    rma_iov.len = bytes;
    rma_iov.key = rkey;

    fi_msg_rma msg{};
    msg.msg_iov = &iov;
    msg.desc = descs;
    msg.iov_count = 1;
    msg.addr = peer_addr;
    msg.rma_iov = &rma_iov;
    msg.rma_iov_count = 1;

    return fi_writemsg(ep, &msg, FI_COMPLETION);
}

static ssize_t PostReadWithCompletion(fid_ep *ep, fi_addr_t peer_addr,
                                      void *local_addr, size_t bytes,
                                      void *desc, uint64_t remote_addr,
                                      uint64_t rkey)
{
    iovec iov{};
    iov.iov_base = local_addr;
    iov.iov_len = bytes;

    void *descs[1] = {desc};

    fi_rma_iov rma_iov{};
    rma_iov.addr = remote_addr;
    rma_iov.len = bytes;
    rma_iov.key = rkey;

    fi_msg_rma msg{};
    msg.msg_iov = &iov;
    msg.desc = descs;
    msg.iov_count = 1;
    msg.addr = peer_addr;
    msg.rma_iov = &rma_iov;
    msg.rma_iov_count = 1;

    return fi_readmsg(ep, &msg, FI_COMPLETION);
}

OFIContext::OFIContext(MPI_Comm comm, const std::string &provider_name)
    : comm(comm), fi(nullptr), fabric(nullptr), domain(nullptr),
      cq(nullptr), av(nullptr), ep(nullptr),
      inject_size(0), mr_mode(0), mr_endpoint(false),
      mr_key_counter(0), cq_size(DEFAULT_CQ_SIZE),
      outstanding(0), live_mr_count(0), freed(false),
      connected_mode(false), pep(nullptr), eq(nullptr)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);

    if(this->rank == 0)
    {
        fprintf(stderr, "[OFI] initializing shared context on %d ranks\n", this->size);
    }

    std::exception_ptr setup_error;
    try
    {
        this->SetupFabric(provider_name);
    }
    catch(...)
    {
        setup_error = std::current_exception();
    }

    int local_setup_success = setup_error ? 0 : 1;
    int all_setup_success = 0;
    MPI_Allreduce(&local_setup_success, &all_setup_success, 1, MPI_INT, MPI_MIN, this->comm);
    if(not all_setup_success)
    {
        if(setup_error)
        {
            try
            {
                std::rethrow_exception(setup_error);
            }
            catch(const std::exception &e)
            {
                fprintf(stderr, "[OFI rank %d] fabric setup failed: %s\n",
                        this->rank, e.what());
            }
            catch(...)
            {
            }
        }
        this->Free();
        throw std::runtime_error("OFIContext: fabric setup failed on at least one rank");
    }

    if(this->connected_mode)
    {
        this->EstablishConnections();
    }
    else
    {
        this->ExchangeAddresses();
    }
}

OFIContext::~OFIContext()
{
    if(not std::uncaught_exceptions() and not this->freed)
    {
        this->Free();
    }
}

static std::string Lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

static std::string ProviderName(const fi_info *info)
{
    if(not info or not info->fabric_attr or not info->fabric_attr->prov_name)
    {
        return "";
    }
    return Lowercase(info->fabric_attr->prov_name);
}

static std::string ProviderBase(const std::string &provider)
{
    std::string base = provider;
    size_t sep = base.find(';');
    if(sep != std::string::npos)
    {
        base = base.substr(0, sep);
    }
    return base;
}

static bool IsCXIProvider(const fi_info *info)
{
    return ProviderBase(ProviderName(info)) == "cxi";
}

static std::string DomainName(const fi_info *info)
{
    if(not info or not info->domain_attr or not info->domain_attr->name)
    {
        return "";
    }
    return info->domain_attr->name;
}

#ifdef STORM_WITH_CXI_PALS_AUTH
struct CXIAuthKeySelection
{
    bool available = false;
    cxi_auth_key key{};
    std::string device_name;
    int profile_count = 0;
    std::string error;
};

static std::string PALSErrorMessage(pals_state_t *state, const char *call)
{
    std::string message = std::string(call) + " failed";
    if(state)
    {
        const char *detail = pals_errmsg(state);
        if(detail and detail[0] != '\0')
        {
            message += ": ";
            message += detail;
        }
    }
    return message;
}

static CXIAuthKeySelection LoadCXIAuthKeyFromPALS(const fi_info *info)
{
    CXIAuthKeySelection selection;

    pals_state_t *state = nullptr;
    pals_rc_t rc = pals_init2(&state);
    if(rc != PALS_OK)
    {
        selection.error = PALSErrorMessage(state, "pals_init2");
        if(state)
        {
            pals_fini(state);
        }
        return selection;
    }

    pals_comm_profile_t *profiles = nullptr;
    int nprofiles = 0;
    rc = pals_get_comm_profiles(state, &profiles, &nprofiles);
    if(rc != PALS_OK)
    {
        selection.error = PALSErrorMessage(state, "pals_get_comm_profiles");
        std::free(profiles);
        pals_fini(state);
        return selection;
    }

    selection.profile_count = nprofiles;
    const std::string domain_name = DomainName(info);
    const pals_comm_profile_t *chosen = nullptr;

    for(int i = 0; i < nprofiles; ++i)
    {
        if(profiles[i].nvnis == 0)
        {
            continue;
        }

        if(domain_name.empty() or domain_name == profiles[i].device_name)
        {
            chosen = &profiles[i];
            break;
        }
    }

    if(not chosen)
    {
        for(int i = 0; i < nprofiles; ++i)
        {
            if(profiles[i].nvnis > 0)
            {
                chosen = &profiles[i];
                break;
            }
        }
    }

    if(chosen)
    {
        selection.available = true;
        selection.key.svc_id = chosen->svc_id;
        selection.key.vni = chosen->vnis[0];
        selection.device_name = chosen->device_name;
    }
    else
    {
        selection.error = "no PALS communication profile with a VNI";
        if(not domain_name.empty())
        {
            selection.error += " for domain ";
            selection.error += domain_name;
        }
    }

    std::free(profiles);
    pals_fini(state);
    return selection;
}
#endif

static void ConfigureCXIAuthKey(fi_info *info, int rank)
{
    if(not IsCXIProvider(info) or not info or not info->domain_attr)
    {
        return;
    }

    if(info->domain_attr->auth_key and info->domain_attr->auth_key_size != 0)
    {
        return;
    }

#ifdef STORM_WITH_CXI_PALS_AUTH
    CXIAuthKeySelection selection = LoadCXIAuthKeyFromPALS(info);
    if(not selection.available)
    {
        if(rank == 0)
        {
            fprintf(stderr, "[OFI] CXI PALS auth key unavailable: %s\n",
                    selection.error.empty() ? "unknown PALS error" : selection.error.c_str());
        }
        return;
    }

    auto *auth_key = static_cast<cxi_auth_key*>(std::malloc(sizeof(cxi_auth_key)));
    if(not auth_key)
    {
        ThrowOFIError("malloc failed for CXI auth key");
    }
    *auth_key = selection.key;

    info->domain_attr->auth_key = reinterpret_cast<uint8_t*>(auth_key);
    info->domain_attr->auth_key_size = sizeof(cxi_auth_key);

    if(rank == 0)
    {
        fprintf(stderr, "[OFI] CXI auth key from PALS: device=%s svc_id=%u vni=%u profiles=%d\n",
                selection.device_name.c_str(),
                static_cast<unsigned>(selection.key.svc_id),
                static_cast<unsigned>(selection.key.vni),
                selection.profile_count);
    }
#else
    if(rank == 0)
    {
        fprintf(stderr, "[OFI] CXI PALS auth-key support is not compiled in; "
                "relying on libfabric environment/default service discovery\n");
    }
#endif
}

static bool IsLayeredProvider(const std::string &provider)
{
    return provider.find(';') != std::string::npos;
}

static bool IsHardwareRDMAProvider(const std::string &provider)
{
    const std::string base = ProviderBase(provider);
    return base == "cxi" or base == "efa" or base == "psm2" or
           base == "psm3" or base == "gni" or
           base == "opx" or base == "mlx";
}

static bool IsInfiniBandVerbsProvider(const std::string &provider)
{
    return ProviderBase(provider) == "verbs";
}

static int HardwareProviderScore(const std::string &provider)
{
    const std::string base = ProviderBase(provider);
    if(base == "cxi") return 100;
    if(base == "efa") return 90;
    if(base == "psm3") return 85;
    if(base == "psm2") return 80;
    if(base == "opx") return 75;
    if(base == "gni") return 70;
    if(base == "mlx") return 65;
    return -1;
}

static std::vector<std::string> DefaultRDMProviderOrder()
{
    return {"cxi", "efa", "psm3", "psm2", "opx", "gni", "mlx", "verbs"};
}

static std::vector<std::string> SplitProviderList(const char *providers)
{
    std::vector<std::string> result;
    if(not providers)
    {
        return result;
    }

    std::string value = providers;
    bool exclude_list = false;
    if(not value.empty() and value[0] == '^')
    {
        exclude_list = true;
        value.erase(value.begin());
    }

    size_t pos = 0;
    while(pos <= value.size())
    {
        size_t comma = value.find(',', pos);
        std::string token = value.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if(not token.empty())
        {
            result.push_back(Lowercase(token));
        }
        if(comma == std::string::npos)
        {
            break;
        }
        pos = comma + 1;
    }

    if(exclude_list)
    {
        std::vector<std::string> included = DefaultRDMProviderOrder();
        included.erase(std::remove_if(included.begin(), included.end(),
                                      [&](const std::string &provider) {
                                          const std::string base = ProviderBase(Lowercase(provider));
                                          return std::find_if(result.begin(), result.end(),
                                                              [&](const std::string &excluded) {
                                                                  return ProviderBase(excluded) == base;
                                                              }) != result.end();
                                      }),
                       included.end());
        return included;
    }

    return result;
}

static std::vector<std::string> RDMProviderProbeOrder(const std::string &provider_name,
                                                       const std::string &exclude_family = "")
{
    if(not provider_name.empty())
    {
        return {provider_name};
    }

    const char *env_provider = std::getenv("FI_PROVIDER");
    if(env_provider and env_provider[0] != '\0')
    {
        std::vector<std::string> providers = SplitProviderList(env_provider);
        if(not exclude_family.empty())
        {
            providers.erase(std::remove_if(providers.begin(), providers.end(),
                                           [&](const std::string &provider) {
                                               return ProviderBase(Lowercase(provider)) == exclude_family;
                                           }),
                            providers.end());
        }
        return providers;
    }

    std::vector<std::string> providers = DefaultRDMProviderOrder();
    if(not exclude_family.empty())
    {
        providers.erase(std::remove_if(providers.begin(), providers.end(),
                                       [&](const std::string &provider) {
                                           return ProviderBase(Lowercase(provider)) == exclude_family;
                                       }),
                        providers.end());
    }
    return providers;
}

static std::vector<std::string> MSGProviderProbeOrder(const std::string &provider_name)
{
    if(not provider_name.empty())
    {
        std::string msg_provider_name = provider_name;
        if(IsInfiniBandVerbsProvider(Lowercase(msg_provider_name)))
        {
            msg_provider_name = "verbs";
        }
        return {msg_provider_name};
    }

    const char *env_provider = std::getenv("FI_PROVIDER");
    if(env_provider and env_provider[0] != '\0')
    {
        std::vector<std::string> providers = SplitProviderList(env_provider);
        for(const std::string &provider : providers)
        {
            if(IsInfiniBandVerbsProvider(provider))
            {
                // A positive core-provider filter such as "verbs" may expose
                // layered providers as well. Always issue the MSG query against
                // the exact native provider name.
                return {"verbs"};
            }
        }
        return {};
    }

    return {"verbs"};
}

static fi_info *QueryOFIInfo(uint64_t caps, fi_ep_type ep_type,
                             uint64_t mode, uint64_t mr_mode,
                             const std::string &provider_name, int &ret)
{
    fi_info *hints = fi_allocinfo();
    if(not hints)
    {
        ret = -FI_ENOMEM;
        return nullptr;
    }

    hints->caps = caps;
    hints->mode = mode;
    hints->ep_attr->type = ep_type;
    if(mr_mode != 0)
    {
        hints->domain_attr->mr_mode = mr_mode;
    }

    if(not provider_name.empty())
    {
        hints->fabric_attr->prov_name = strdup(provider_name.c_str());
    }

    fi_info *info_list = nullptr;
    ret = fi_getinfo(FI_VERSION(1, 6), nullptr, nullptr, 0, hints, &info_list);
    fi_freeinfo(hints);
    if(ret != 0)
    {
        return nullptr;
    }
    return info_list;
}

static void ReportProviderQueryFailure(int rank, const char *query_name,
                                       const std::string &provider_name, int ret)
{
    const char *env_provider = std::getenv("FI_PROVIDER");
    const char *filter = (env_provider and env_provider[0] != '\0') ?
        env_provider : "<unset>";
    const char *requested = provider_name.empty() ?
        "<auto>" : provider_name.c_str();

    if(ret != 0)
    {
        fprintf(stderr,
                "[OFI rank %d] %s discovery failed: requested=%s "
                "FI_PROVIDER=%s fi_getinfo=%d (%s)\n",
                rank, query_name, requested, filter, ret, fi_strerror(-ret));
    }
    else
    {
        fprintf(stderr,
                "[OFI rank %d] %s discovery returned interfaces, but none "
                "matched the required exact native provider; requested=%s "
                "FI_PROVIDER=%s\n",
                rank, query_name, requested, filter);
    }
}

static fi_info *ChooseBestRDMProvider(fi_info *list, const std::string &exclude_family = "")
{
    fi_info *best = nullptr;
    int best_score = -1;

    for(fi_info *cur = list; cur != nullptr; cur = cur->next)
    {
        if(not cur->domain_attr or not cur->ep_attr)
        {
            continue;
        }

        std::string pname = ProviderName(cur);
        std::string family = ProviderBase(pname);

        // Utility providers such as verbs;ofi_rxd and verbs;ofi_rxm do not
        // provide the native ordering and progress semantics required here.
        // Native verbs is handled separately through FI_EP_MSG/RC.
        if(IsLayeredProvider(pname) or family == "verbs")
        {
            continue;
        }

        if((cur->caps & (FI_RMA | FI_ATOMIC)) != (FI_RMA | FI_ATOMIC))
        {
            continue;
        }

        if(cur->ep_attr->type != FI_EP_RDM)
        {
            continue;
        }

        if(not exclude_family.empty())
        {
            if(family == exclude_family) continue;
        }

        if(not IsHardwareRDMAProvider(pname))
        {
            continue;
        }

        int score = HardwareProviderScore(pname);

        bool virt_addr = (cur->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0;
        bool prov_key = (cur->domain_attr->mr_mode & FI_MR_PROV_KEY) != 0;
        bool endpoint = (cur->domain_attr->mr_mode & FI_MR_ENDPOINT) != 0;
        if(prov_key) score += 20;
        if(endpoint) score += 5;
        if(virt_addr) score += 10;

        if(score > best_score)
        {
            best_score = score;
            best = cur;
        }
    }

    return best;
}

static fi_info *FindBestRDMProviderInfo(const std::string &provider_name,
                                        const std::string &exclude_family,
                                        int &last_ret)
{
    last_ret = -FI_ENODATA;
    for(const std::string &provider : RDMProviderProbeOrder(provider_name, exclude_family))
    {
        int ret = 0;
        fi_info *info_list = QueryOFIInfo(FI_RMA | FI_ATOMIC,
                                          FI_EP_RDM,
                                          0,
                                          FI_MR_ALLOCATED | FI_MR_PROV_KEY |
                                              FI_MR_ENDPOINT,
                                          provider, ret);
        if(ret != 0)
        {
            last_ret = ret;
            continue;
        }

        fi_info *chosen = ChooseBestRDMProvider(info_list, exclude_family);
        fi_info *result = chosen ? fi_dupinfo(chosen) : nullptr;
        fi_freeinfo(info_list);
        if(result)
        {
            last_ret = 0;
            return result;
        }
    }
    return nullptr;
}

static fi_info *ChooseMSGProvider(fi_info *list)
{
    fi_info *best = nullptr;
    int best_score = -1;

    for(fi_info *cur = list; cur != nullptr; cur = cur->next)
    {
        if(not cur->domain_attr or not cur->ep_attr)
        {
            continue;
        }

        std::string pname = ProviderName(cur);
        int score = 0;
        if(pname == "verbs") score = 100;
        else continue;

        if((cur->caps & (FI_RMA | FI_ATOMIC)) != (FI_RMA | FI_ATOMIC))
        {
            continue;
        }

        if(cur->ep_attr->type != FI_EP_MSG)
        {
            continue;
        }

        bool virt_addr = (cur->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0;
        bool prov_key = (cur->domain_attr->mr_mode & FI_MR_PROV_KEY) != 0;
        if(prov_key) score += 20;
        if(virt_addr) score += 10;

        if(score > best_score)
        {
            best_score = score;
            best = cur;
        }
    }

    return best;
}

static fi_info *FindBestMSGProviderInfo(const std::string &provider_name, int &last_ret)
{
    last_ret = -FI_ENODATA;
    for(const std::string &provider : MSGProviderProbeOrder(provider_name))
    {
        int ret = 0;
        fi_info *info_list = QueryOFIInfo(FI_RMA | FI_ATOMIC,
                                          FI_EP_MSG,
                                          FI_RX_CQ_DATA,
                                          FI_MR_LOCAL | FI_MR_VIRT_ADDR |
                                              FI_MR_ALLOCATED | FI_MR_PROV_KEY,
                                          provider, ret);
        if(ret != 0)
        {
            last_ret = ret;
            continue;
        }

        fi_info *chosen = ChooseMSGProvider(info_list);
        fi_info *result = chosen ? fi_dupinfo(chosen) : nullptr;
        fi_freeinfo(info_list);
        if(result)
        {
            last_ret = 0;
            return result;
        }
    }
    return nullptr;
}

// CXI discovery and domain creation can contend among processes sharing a NIC.
// Serialize only within each node; world-rank serialization makes startup O(P).
template<typename Call>
static auto RunNodeSerializedOFICall(MPI_Comm comm, const Call &call) -> decltype(call())
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if(not mpi_initialized or comm == MPI_COMM_NULL)
    {
        return call();
    }

    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);
    if(comm_size <= 1)
    {
        return call();
    }

#if MPI_VERSION < 3
    return call();
#else
    using Result = decltype(call());
    MPI_Comm node_comm = MPI_COMM_NULL;
    int mpi_ret = MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
                                      MPI_INFO_NULL, &node_comm);
    int local_split_success =
        (mpi_ret == MPI_SUCCESS and node_comm != MPI_COMM_NULL) ? 1 : 0;
    int all_split_success = 0;
    MPI_Allreduce(&local_split_success, &all_split_success, 1, MPI_INT, MPI_MIN, comm);
    if(not all_split_success)
    {
        if(node_comm != MPI_COMM_NULL)
        {
            MPI_Comm_free(&node_comm);
        }
        throw std::runtime_error(
            "OFIContext: MPI_Comm_split_type failed during serialized OFI setup");
    }

    int node_rank = 0;
    int node_size = 1;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    Result result{};
    std::exception_ptr error;

    for(int turn = 0; turn < node_size; ++turn)
    {
        if(node_rank == turn)
        {
            try
            {
                result = call();
            }
            catch(...)
            {
                error = std::current_exception();
            }
        }
        MPI_Barrier(node_comm);
    }

    MPI_Comm_free(&node_comm);

    int local_exception = error ? 1 : 0;
    int any_exception = 0;
    MPI_Allreduce(&local_exception, &any_exception, 1, MPI_INT, MPI_MAX, comm);
    if(any_exception)
    {
        if(error)
        {
            try
            {
                std::rethrow_exception(error);
            }
            catch(const std::exception &e)
            {
                int rank = 0;
                MPI_Comm_rank(comm, &rank);
                fprintf(stderr, "[OFI rank %d] serialized setup call failed: %s\n",
                        rank, e.what());
            }
            catch(...)
            {
            }
        }
        throw std::runtime_error(
            "OFIContext: serialized OFI setup failed on at least one rank");
    }
    return result;
#endif
}

static bool AllRanksTrue(MPI_Comm comm, bool value)
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if(not mpi_initialized or comm == MPI_COMM_NULL)
    {
        return value;
    }

    int local_value = value ? 1 : 0;
    int global_value = 0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN, comm);
    return global_value != 0;
}

struct CollectiveOFIStatus
{
    bool success;
    int error;
    int rank;
};

static CollectiveOFIStatus GetCollectiveOFIStatus(MPI_Comm comm, int local_error)
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if(not mpi_initialized or comm == MPI_COMM_NULL)
    {
        return {local_error == 0, local_error, local_error == 0 ? -1 : 0};
    }

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    int local_failed_rank = local_error == 0 ? INT_MAX : rank;
    int first_failed_rank = INT_MAX;
    MPI_Allreduce(&local_failed_rank, &first_failed_rank, 1, MPI_INT, MPI_MIN, comm);
    if(first_failed_rank == INT_MAX)
    {
        return {true, 0, -1};
    }

    int first_error = rank == first_failed_rank ? local_error : 0;
    MPI_Bcast(&first_error, 1, MPI_INT, first_failed_rank, comm);
    return {false, first_error, first_failed_rank};
}

static std::string OFIProviderComponent(const std::string &provider)
{
    std::string component = provider;
    size_t sep = component.find(';');
    if(sep != std::string::npos)
    {
        component = component.substr(0, sep);
    }

    if(component == "verbs")
    {
        return "verbs/ibv";
    }
    return component;
}

bool OFIContext::HasUsableProvider(const std::string &provider_name, MPI_Comm comm)
{
    int ret = 0;
    fi_info *info = RunNodeSerializedOFICall(comm, [&]() {
        return FindBestRDMProviderInfo(provider_name, "", ret);
    });
    bool all_have_rdm = AllRanksTrue(comm, info != nullptr);
    if(info and not all_have_rdm)
    {
        fi_freeinfo(info);
        info = nullptr;
    }
    if(all_have_rdm)
    {
        fi_freeinfo(info);
        return true;
    }

    info = RunNodeSerializedOFICall(comm, [&]() {
        return FindBestMSGProviderInfo(provider_name, ret);
    });
    bool all_have_msg = AllRanksTrue(comm, info != nullptr);
    if(info and not all_have_msg)
    {
        fi_freeinfo(info);
        info = nullptr;
    }
    if(all_have_msg)
    {
        fi_freeinfo(info);
        return true;
    }

    return false;
}

fid_ep *OFIContext::CreateBoundEndpoint(fi_info *ep_info)
{
    fid_ep *new_ep = nullptr;
    int ret = fi_endpoint(this->domain, ep_info, &new_ep, nullptr);
    if(ret != 0)
    {
        ThrowOFIError("fi_endpoint failed", -ret);
    }

    ret = fi_ep_bind(new_ep, &this->cq->fid,
                     FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION);
    if(ret != 0)
    {
        ret = fi_ep_bind(new_ep, &this->cq->fid, FI_TRANSMIT | FI_RECV);
    }
    if(ret != 0)
    {
        fi_close(&new_ep->fid);
        ThrowOFIError("fi_ep_bind(cq) failed", -ret);
    }

    ret = fi_ep_bind(new_ep, &this->eq->fid, 0);
    if(ret != 0)
    {
        fi_close(&new_ep->fid);
        ThrowOFIError("fi_ep_bind(eq) failed", -ret);
    }

    ret = fi_enable(new_ep);
    if(ret != 0)
    {
        fi_close(&new_ep->fid);
        ThrowOFIError("fi_enable failed", -ret);
    }

    return new_ep;
}

void OFIContext::SetupFabric(const std::string &provider_name)
{
    auto closeFabricDomain = [this]()
    {
        if(this->domain)
        {
            fi_close(&this->domain->fid);
            this->domain = nullptr;
        }
        if(this->fabric)
        {
            fi_close(&this->fabric->fid);
            this->fabric = nullptr;
        }
    };

    auto openFabricDomain = [this, &closeFabricDomain]()
    {
        int local_error = RunNodeSerializedOFICall(this->comm, [this]()
        {
            int error = fi_fabric(this->fi->fabric_attr, &this->fabric, nullptr);
            if(error == 0)
            {
                try
                {
                    ConfigureCXIAuthKey(this->fi, this->rank);
                }
                catch(const std::exception &e)
                {
                    fprintf(stderr, "[OFI rank %d] provider authorization setup failed: %s\n",
                            this->rank, e.what());
                    error = -FI_EOTHER;
                }
            }
            if(error == 0)
            {
                error = fi_domain(this->fabric, this->fi, &this->domain, nullptr);
            }
            return error;
        });

        CollectiveOFIStatus status = GetCollectiveOFIStatus(this->comm, local_error);
        if(not status.success)
        {
            closeFabricDomain();
        }
        return status;
    };

    int ret = 0;
    this->fi = RunNodeSerializedOFICall(this->comm, [&]() {
        return FindBestRDMProviderInfo(provider_name, "", ret);
    });
    bool all_have_rdm = AllRanksTrue(this->comm, this->fi != nullptr);
    if(this->fi and not all_have_rdm)
    {
        fi_freeinfo(this->fi);
        this->fi = nullptr;
    }
    if(all_have_rdm)
    {
        std::string failed_provider = this->fi->fabric_attr->prov_name ?
            this->fi->fabric_attr->prov_name : "";
        std::string exclude_family = ProviderBase(Lowercase(failed_provider));
        CollectiveOFIStatus status = openFabricDomain();
        if(not status.success)
        {
            if(this->rank == 0)
            {
                fprintf(stderr,
                        "[OFI] provider %s: fabric/domain setup failed on rank %d (%s), "
                        "retrying without it\n",
                        failed_provider.c_str(), status.rank,
                        status.error == 0 ? "unknown error" : fi_strerror(-status.error));
                if(exclude_family == "cxi")
                {
                    fprintf(stderr, "[OFI] CXI domain creation requires a valid service/VNI "
                            "authorization key. CXI may obtain it from SLINGSHOT_* "
                            "environment variables, configured UID/GID services, an unrestricted "
                            "service, or FI_CXI_DEFAULT_VNI; SLINGSHOT_VNIS being unset is not "
                            "by itself an error.\n");
                    fprintf(stderr, "[OFI] EasyRMA tries libpals first. If the preceding "
                            "PALS line says PALS_APINFO is unset, this process was not launched "
                            "inside a PALS/Cray-Slurm job step that exposes CXI communication "
                            "profiles.\n");
                }
            }

            fi_freeinfo(this->fi);
            this->fi = nullptr;

            this->fi = RunNodeSerializedOFICall(this->comm, [&]() {
                return FindBestRDMProviderInfo(provider_name, exclude_family, ret);
            });
            bool all_have_fallback = AllRanksTrue(this->comm, this->fi != nullptr);
            if(this->fi and not all_have_fallback)
            {
                fi_freeinfo(this->fi);
                this->fi = nullptr;
            }
            if(not all_have_fallback)
            {
                ThrowOFIError("no fallback provider available after excluding " + exclude_family);
            }

            status = openFabricDomain();
            if(not status.success)
            {
                fi_freeinfo(this->fi);
                this->fi = nullptr;
                ThrowOFIError(
                    "fallback fabric/domain setup failed on rank " + std::to_string(status.rank),
                    status.error == 0 ? 0 : -status.error);
            }
        }

        this->connected_mode = false;
    }
    else
    {
        this->fi = RunNodeSerializedOFICall(this->comm, [&]() {
            return FindBestMSGProviderInfo(provider_name, ret);
        });
        bool local_have_msg = this->fi != nullptr;
        if(not local_have_msg)
        {
            ReportProviderQueryFailure(this->rank, "native verbs FI_EP_MSG",
                                       provider_name, ret);
        }
        bool all_have_msg = AllRanksTrue(this->comm, local_have_msg);
        if(this->fi and not all_have_msg)
        {
            fi_freeinfo(this->fi);
            this->fi = nullptr;
        }
        if(not all_have_msg)
        {
            ThrowOFIError("no suitable hardware OFI provider found (need cxi/efa/psm/gni/opx/mlx RDM or verbs MSG; refusing tcp/sockets)");
        }

        CollectiveOFIStatus status = openFabricDomain();
        if(not status.success)
        {
            fi_freeinfo(this->fi);
            this->fi = nullptr;
            ThrowOFIError(
                "MSG fabric/domain setup failed on rank " + std::to_string(status.rank),
                status.error == 0 ? 0 : -status.error);
        }

        this->connected_mode = true;
    }

    this->mr_mode = this->fi->domain_attr->mr_mode;
    this->mr_endpoint = (this->mr_mode & FI_MR_ENDPOINT) != 0;
    this->inject_size = this->fi->tx_attr->inject_size;

    if(this->rank == 0)
    {
        std::string provider = this->fi->fabric_attr->prov_name ?
            this->fi->fabric_attr->prov_name : "unknown";
        std::string component = OFIProviderComponent(provider);
        fprintf(stderr, "[OFI] component: %s (provider: %s, mode: %s)\n",
                component.c_str(), provider.c_str(),
                this->connected_mode ? "MSG/RC" : "RDM");
        fprintf(stderr, "[OFI] mr_mode=0x%x prov_key=%d endpoint=%d virt_addr=%d "
                "mr_key_size=%zu mr_cnt=%zu\n",
                static_cast<unsigned>(this->fi->domain_attr->mr_mode),
                !!(this->fi->domain_attr->mr_mode & FI_MR_PROV_KEY),
                !!(this->fi->domain_attr->mr_mode & FI_MR_ENDPOINT),
                !!(this->fi->domain_attr->mr_mode & FI_MR_VIRT_ADDR),
                this->fi->domain_attr->mr_key_size,
                this->fi->domain_attr->mr_cnt);
    }

    fi_cq_attr cq_attr{};
    cq_attr.size = this->cq_size;
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    ret = fi_cq_open(this->domain, &cq_attr, &this->cq, nullptr);
    if(ret != 0)
    {
        ThrowOFIError("fi_cq_open failed", -ret);
    }

    fi_eq_attr eq_attr{};
    eq_attr.size = this->size * 2;
    eq_attr.wait_obj = FI_WAIT_UNSPEC;
    ret = fi_eq_open(this->fabric, &eq_attr, &this->eq, nullptr);
    if(ret != 0)
    {
        ThrowOFIError("fi_eq_open failed", -ret);
    }

    if(this->connected_mode)
    {
        ret = fi_passive_ep(this->fabric, this->fi, &this->pep, nullptr);
        if(ret != 0)
        {
            ThrowOFIError("fi_passive_ep failed", -ret);
        }

        ret = fi_pep_bind(this->pep, &this->eq->fid, 0);
        if(ret != 0)
        {
            ThrowOFIError("fi_pep_bind(eq) failed", -ret);
        }

        ret = fi_listen(this->pep);
        if(ret != 0)
        {
            ThrowOFIError("fi_listen failed", -ret);
        }
    }
    else
    {
        fi_av_attr av_attr{};
        av_attr.type = FI_AV_MAP;
        av_attr.count = this->size;
        ret = fi_av_open(this->domain, &av_attr, &this->av, nullptr);
        if(ret != 0)
        {
            ThrowOFIError("fi_av_open failed", -ret);
        }

        ret = fi_endpoint(this->domain, this->fi, &this->ep, nullptr);
        if(ret != 0)
        {
            ThrowOFIError("fi_endpoint failed", -ret);
        }

        ret = fi_ep_bind(this->ep, &this->cq->fid,
                         FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION);
        if(ret != 0)
        {
            ret = fi_ep_bind(this->ep, &this->cq->fid, FI_TRANSMIT | FI_RECV);
        }
        if(ret != 0)
        {
            ThrowOFIError("fi_ep_bind(cq) failed", -ret);
        }

        ret = fi_ep_bind(this->ep, &this->av->fid, 0);
        if(ret != 0)
        {
            ThrowOFIError("fi_ep_bind(av) failed", -ret);
        }

        ret = fi_enable(this->ep);
        if(ret != 0)
        {
            ThrowOFIError("fi_enable failed", -ret);
        }
    }
}

void OFIContext::ExchangeAddresses()
{
    size_t addrlen = 0;
    int ret = fi_getname(&this->ep->fid, nullptr, &addrlen);
    int local_error = (ret == 0 or ret == -FI_ETOOSMALL) ? 0 : ret;
    CollectiveOFIStatus status = GetCollectiveOFIStatus(this->comm, local_error);
    if(not status.success)
    {
        ThrowOFIError("fi_getname(size) failed on rank " + std::to_string(status.rank),
                      status.error == 0 ? 0 : -status.error);
    }

    unsigned long long local_addrlen = static_cast<unsigned long long>(addrlen);
    unsigned long long min_addrlen = 0;
    unsigned long long max_addrlen = 0;
    MPI_Allreduce(&local_addrlen, &min_addrlen, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, this->comm);
    MPI_Allreduce(&local_addrlen, &max_addrlen, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, this->comm);
    if(min_addrlen != max_addrlen or max_addrlen > static_cast<unsigned long long>(INT_MAX))
    {
        throw std::runtime_error("OFIContext: endpoint address lengths differ across ranks");
    }

    std::vector<char> local_addr(addrlen);
    ret = fi_getname(&this->ep->fid, local_addr.data(), &addrlen);
    status = GetCollectiveOFIStatus(this->comm, ret);
    if(not status.success)
    {
        ThrowOFIError("fi_getname failed on rank " + std::to_string(status.rank),
                      status.error == 0 ? 0 : -status.error);
    }

    std::vector<char> all_addrs(this->size * addrlen);
    MPI_Allgather(local_addr.data(), static_cast<int>(addrlen), MPI_BYTE,
                  all_addrs.data(), static_cast<int>(addrlen), MPI_BYTE, this->comm);

    this->peer_addrs.resize(this->size);
    int inserted = fi_av_insert(this->av, all_addrs.data(), this->size,
                                this->peer_addrs.data(), 0, nullptr);
    local_error = inserted == this->size ? 0 : (inserted < 0 ? inserted : -FI_EOTHER);
    status = GetCollectiveOFIStatus(this->comm, local_error);
    if(not status.success)
    {
        ThrowOFIError("fi_av_insert failed on rank " + std::to_string(status.rank),
                      status.error == 0 ? 0 : -status.error);
    }

    this->peer_connected.assign(this->size, true);
}

void OFIContext::EstablishConnections()
{
    this->peer_eps.assign(this->size, nullptr);

    size_t addrlen = 0;
    fi_getname(&this->pep->fid, nullptr, &addrlen);

    std::vector<char> local_addr(addrlen);
    int ret = fi_getname(&this->pep->fid, local_addr.data(), &addrlen);
    if(ret != 0)
    {
        ThrowOFIError("fi_getname(pep) failed", -ret);
    }

    std::vector<char> all_addrs(this->size * addrlen);
    MPI_Allgather(local_addr.data(), static_cast<int>(addrlen), MPI_BYTE,
                  all_addrs.data(), static_cast<int>(addrlen), MPI_BYTE, this->comm);

    for(int peer = 0; peer < this->size; ++peer)
    {
        if(peer == this->rank) continue;
        if(this->rank < peer)
        {
            fid_ep *active_ep = this->CreateBoundEndpoint(this->fi);
            ret = fi_connect(active_ep, all_addrs.data() + peer * addrlen,
                             &this->rank, sizeof(this->rank));
            if(ret != 0)
            {
                fi_close(&active_ep->fid);
                ThrowOFIError("fi_connect to peer " + std::to_string(peer) + " failed", -ret);
            }
            this->peer_eps[peer] = active_ep;
        }
    }

    int total_connections = this->size - 1;
    int established = 0;

    std::vector<char> event_buffer(sizeof(fi_eq_cm_entry) + sizeof(int));

    while(established < total_connections)
    {
        uint32_t event = 0;
        std::fill(event_buffer.begin(), event_buffer.end(), 0);
        auto *entry = reinterpret_cast<fi_eq_cm_entry*>(event_buffer.data());
        ssize_t rd = fi_eq_sread(this->eq, &event, entry, event_buffer.size(), 5000, 0);

        if(rd == -FI_EAGAIN)
        {
            continue;
        }

        if(rd < 0)
        {
            fi_eq_err_entry err{};
            fi_eq_readerr(this->eq, &err, 0);
            ThrowOFIError("EQ error during connection setup (err=" +
                          std::to_string(err.err) + ", prov_errno=" +
                          std::to_string(err.prov_errno) + ")");
        }

        if(event == FI_CONNREQ)
        {
            int peer_rank = -1;
            if(rd >= static_cast<ssize_t>(sizeof(fi_eq_cm_entry) + sizeof(int)))
            {
                std::memcpy(&peer_rank, entry->data, sizeof(int));
            }

            if(peer_rank < 0 or peer_rank >= this->size)
            {
                if(entry->info) fi_freeinfo(entry->info);
                ThrowOFIError("CONNREQ with invalid peer rank: " + std::to_string(peer_rank));
            }

            fid_ep *passive_ep = this->CreateBoundEndpoint(entry->info);
            ret = fi_accept(passive_ep, nullptr, 0);
            if(ret != 0)
            {
                fi_close(&passive_ep->fid);
                if(entry->info) fi_freeinfo(entry->info);
                ThrowOFIError("fi_accept from peer " + std::to_string(peer_rank) + " failed", -ret);
            }
            this->peer_eps[peer_rank] = passive_ep;
            if(entry->info) fi_freeinfo(entry->info);
        }
        else if(event == FI_CONNECTED)
        {
            established++;
        }
    }

    this->peer_connected.assign(this->size, true);
    this->peer_connected[this->rank] = false;

    MPI_Barrier(this->comm);

    if(this->rank == 0)
    {
        fprintf(stderr, "[OFI] MSG mode: all %d connections established\n", total_connections);
    }
}

fid_ep *OFIContext::GetEP(int target_rank) const
{
    return this->ResolveEP(target_rank);
}

fi_addr_t OFIContext::GetPeerAddr(int world_rank) const
{
    return this->ResolveAddr(world_rank);
}

fid_mr *OFIContext::RegisterMemory(void *buf, size_t bytes, uint64_t access)
{
    fid_mr *mr = nullptr;

    uint64_t requested_key = 0;
    bool prov_key = (this->mr_mode & FI_MR_PROV_KEY) != 0;
    if(not prov_key)
    {
        requested_key = this->mr_key_counter++;
    }

    int ret = fi_mr_reg(this->domain, buf, bytes, access,
                        0, requested_key, 0, &mr, nullptr);
    if(ret != 0)
    {
        char access_hex[32];
        snprintf(access_hex, sizeof(access_hex), "0x%llx", static_cast<unsigned long long>(access));
        std::string detail = "fi_mr_reg failed (bytes=" + std::to_string(bytes) +
            ", access=" + access_hex +
            ", key=" + std::to_string(requested_key) +
            ", prov_key=" + std::to_string(prov_key) +
            ", mr_key_counter=" + std::to_string(this->mr_key_counter) +
            ", live_mrs=" + std::to_string(this->live_mr_count) + ")";
        ThrowOFIError(detail, -ret);
    }

    this->BindMemoryToEndpoints(mr);

    this->live_mr_count++;
    return mr;
}

void OFIContext::BindMemoryToEndpoint(fid_mr *mr, fid_ep *target_ep)
{
    if(not target_ep)
    {
        return;
    }

    int ret = fi_mr_bind(mr, &target_ep->fid, 0);
    if(ret != 0)
    {
        fi_close(&mr->fid);
        ThrowOFIError("fi_mr_bind failed", -ret);
    }
}

void OFIContext::BindMemoryToEndpoints(fid_mr *mr)
{
    if(not this->mr_endpoint)
    {
        return;
    }

    if(this->connected_mode)
    {
        for(fid_ep *peer_ep : this->peer_eps)
        {
            this->BindMemoryToEndpoint(mr, peer_ep);
        }
    }
    else
    {
        this->BindMemoryToEndpoint(mr, this->ep);
    }

    int ret = fi_mr_enable(mr);
    if(ret != 0)
    {
        fi_close(&mr->fid);
        ThrowOFIError("fi_mr_enable failed", -ret);
    }
}

void OFIContext::DeregisterMemory(fid_mr *mr)
{
    if(not mr)
    {
        return;
    }

    int ret = fi_close(&mr->fid);
    if(ret != 0)
    {
        fprintf(stderr, "[OFI] WARNING: fi_close(mr) failed: %s\n", fi_strerror(-ret));
    }
    else
    {
        this->live_mr_count--;
    }
}

fid_ep *OFIContext::ResolveEP(int target_rank) const
{
    return this->connected_mode ? this->peer_eps[target_rank] : this->ep;
}

fi_addr_t OFIContext::ResolveAddr(int target_rank) const
{
    return this->connected_mode ? FI_ADDR_UNSPEC : this->peer_addrs[target_rank];
}

void OFIContext::EnsureCQSpace()
{
    if(this->outstanding >= this->cq_size / 2)
    {
        this->DrainCompletions();
    }
}

void OFIContext::PostRDMAWrite(int target_rank, const void *local_addr, size_t bytes,
                               void *desc, uint64_t remote_addr, uint64_t rkey, bool signaled)
{
    (void)signaled;
    constexpr size_t MAX_CHUNK = 1ULL << 30;
    const uint8_t *src = static_cast<const uint8_t*>(local_addr);
    fid_ep *target_ep = this->ResolveEP(target_rank);
    fi_addr_t addr = this->ResolveAddr(target_rank);

    while(bytes > MAX_CHUNK)
    {
        this->EnsureCQSpace();
        ssize_t ret;
        do
        {
            ret = PostWriteWithCompletion(target_ep, addr,
                                          src, MAX_CHUNK, desc, remote_addr, rkey);
            if(ret == -FI_EAGAIN)
            {
                this->PollCompletions(1);
            }
        }
        while(ret == -FI_EAGAIN);

        if(ret != 0)
        {
            ThrowOFIError("fi_write failed", static_cast<int>(-ret));
        }
        this->outstanding++;

        src += MAX_CHUNK;
        remote_addr += MAX_CHUNK;
        bytes -= MAX_CHUNK;
    }

    this->EnsureCQSpace();

    ssize_t ret;
    do
    {
        ret = PostWriteWithCompletion(target_ep, addr,
                                      src, bytes, desc, remote_addr, rkey);
        if(ret == -FI_EAGAIN)
        {
            this->PollCompletions(1);
        }
    }
    while(ret == -FI_EAGAIN);

    if(ret != 0)
    {
        ThrowOFIError("fi_write failed", static_cast<int>(-ret));
    }
    this->outstanding++;
}

void OFIContext::PostInjectWrite(int target_rank, const void *local_addr, size_t bytes,
                                 uint64_t remote_addr, uint64_t rkey)
{
    fid_ep *target_ep = this->ResolveEP(target_rank);
    fi_addr_t addr = this->ResolveAddr(target_rank);

    ssize_t ret;
    do
    {
        ret = fi_inject_write(target_ep, local_addr, bytes,
                              addr, remote_addr, rkey);
        if(ret == -FI_EAGAIN)
        {
            this->PollCompletions(1);
        }
    }
    while(ret == -FI_EAGAIN);

    if(ret != 0)
    {
        ThrowOFIError("fi_inject_write failed", static_cast<int>(-ret));
    }
}

void OFIContext::PostRDMARead(int target_rank, void *local_addr, size_t bytes,
                              void *desc, uint64_t remote_addr, uint64_t rkey, bool signaled)
{
    (void)signaled;
    constexpr size_t MAX_CHUNK = 1ULL << 30;
    uint8_t *dst = static_cast<uint8_t*>(local_addr);
    fid_ep *target_ep = this->ResolveEP(target_rank);
    fi_addr_t addr = this->ResolveAddr(target_rank);

    while(bytes > MAX_CHUNK)
    {
        this->EnsureCQSpace();
        ssize_t ret;
        do
        {
            ret = PostReadWithCompletion(target_ep, addr,
                                         dst, MAX_CHUNK, desc, remote_addr, rkey);
            if(ret == -FI_EAGAIN)
            {
                this->PollCompletions(1);
            }
        }
        while(ret == -FI_EAGAIN);

        if(ret != 0)
        {
            ThrowOFIError("fi_read failed", static_cast<int>(-ret));
        }
        this->outstanding++;

        dst += MAX_CHUNK;
        remote_addr += MAX_CHUNK;
        bytes -= MAX_CHUNK;
    }

    this->EnsureCQSpace();

    ssize_t ret;
    do
    {
        ret = PostReadWithCompletion(target_ep, addr,
                                     dst, bytes, desc, remote_addr, rkey);
        if(ret == -FI_EAGAIN)
        {
            this->PollCompletions(1);
        }
    }
    while(ret == -FI_EAGAIN);

    if(ret != 0)
    {
        ThrowOFIError("fi_read failed", static_cast<int>(-ret));
    }
    this->outstanding++;
}

void OFIContext::PostFencedRDMARead(int target_rank, void *local_addr, size_t bytes,
                                    void *desc, uint64_t remote_addr, uint64_t rkey)
{
    this->EnsureCQSpace();
    fid_ep *target_ep = this->ResolveEP(target_rank);
    fi_addr_t addr = this->ResolveAddr(target_rank);

    iovec iov{};
    iov.iov_base = local_addr;
    iov.iov_len = bytes;

    void *descs[1] = {desc};

    fi_rma_iov rma_iov{};
    rma_iov.addr = remote_addr;
    rma_iov.len = bytes;
    rma_iov.key = rkey;

    fi_msg_rma msg{};
    msg.msg_iov = &iov;
    msg.desc = descs;
    msg.iov_count = 1;
    msg.addr = addr;
    msg.rma_iov = &rma_iov;
    msg.rma_iov_count = 1;

    ssize_t ret;
    do
    {
        ret = fi_readmsg(target_ep, &msg, FI_FENCE | FI_COMPLETION);
        if(ret == -FI_EAGAIN)
        {
            this->PollCompletions(1);
        }
    }
    while(ret == -FI_EAGAIN);

    if(ret != 0)
    {
        ThrowOFIError("fi_readmsg (fenced) failed", static_cast<int>(-ret));
    }
    this->outstanding++;
}

void OFIContext::PostAtomicCAS(int target_rank,
                               const void *swap_val, void *swap_desc,
                               const void *compare_val, void *compare_desc,
                               void *result, void *result_desc,
                               uint64_t remote_addr, uint64_t rkey,
                               enum fi_datatype dtype, bool signaled)
{
    (void)signaled;
    this->EnsureCQSpace();
    fid_ep *target_ep = this->ResolveEP(target_rank);
    fi_addr_t addr = this->ResolveAddr(target_rank);

    fi_ioc swap_iov{};
    swap_iov.addr = const_cast<void*>(swap_val);
    swap_iov.count = 1;
    void *swap_descs[1] = {swap_desc};

    fi_ioc compare_iov{};
    compare_iov.addr = const_cast<void*>(compare_val);
    compare_iov.count = 1;
    void *compare_descs[1] = {compare_desc};

    fi_ioc result_iov{};
    result_iov.addr = result;
    result_iov.count = 1;
    void *result_descs[1] = {result_desc};

    fi_rma_ioc rma_iov{};
    rma_iov.addr = remote_addr;
    rma_iov.count = 1;
    rma_iov.key = rkey;

    fi_msg_atomic msg{};
    msg.msg_iov = &swap_iov;
    msg.desc = swap_descs;
    msg.iov_count = 1;
    msg.addr = addr;
    msg.rma_iov = &rma_iov;
    msg.rma_iov_count = 1;
    msg.datatype = dtype;
    msg.op = FI_CSWAP;

    ssize_t ret;
    do
    {
        ret = fi_compare_atomicmsg(target_ep, &msg,
                                    &compare_iov, compare_descs, 1,
                                    &result_iov, result_descs, 1,
                                    FI_FENCE | FI_COMPLETION);
        if(ret == -FI_EAGAIN)
        {
            this->PollCompletions(1);
        }
    }
    while(ret == -FI_EAGAIN);

    if(ret != 0)
    {
        ThrowOFIError("fi_compare_atomic (CAS) failed", static_cast<int>(-ret));
    }
    this->outstanding++;
}

void OFIContext::PostAtomicFetchAdd(int target_rank,
                                    const void *addend, void *addend_desc,
                                    void *result, void *result_desc,
                                    uint64_t remote_addr, uint64_t rkey,
                                    enum fi_datatype dtype, bool signaled)
{
    (void)signaled;
    this->EnsureCQSpace();
    fid_ep *target_ep = this->ResolveEP(target_rank);
    fi_addr_t addr = this->ResolveAddr(target_rank);

    fi_ioc addend_iov{};
    addend_iov.addr = const_cast<void*>(addend);
    addend_iov.count = 1;
    void *addend_descs[1] = {addend_desc};

    fi_ioc result_iov{};
    result_iov.addr = result;
    result_iov.count = 1;
    void *result_descs[1] = {result_desc};

    fi_rma_ioc rma_iov{};
    rma_iov.addr = remote_addr;
    rma_iov.count = 1;
    rma_iov.key = rkey;

    fi_msg_atomic msg{};
    msg.msg_iov = &addend_iov;
    msg.desc = addend_descs;
    msg.iov_count = 1;
    msg.addr = addr;
    msg.rma_iov = &rma_iov;
    msg.rma_iov_count = 1;
    msg.datatype = dtype;
    msg.op = FI_SUM;

    ssize_t ret;
    do
    {
        ret = fi_fetch_atomicmsg(target_ep, &msg,
                                  &result_iov, result_descs, 1,
                                  FI_FENCE | FI_COMPLETION);
        if(ret == -FI_EAGAIN)
        {
            this->PollCompletions(1);
        }
    }
    while(ret == -FI_EAGAIN);

    if(ret != 0)
    {
        ThrowOFIError("fi_fetch_atomic (FetchAdd) failed", static_cast<int>(-ret));
    }
    this->outstanding++;
}

int OFIContext::PollCompletions(int max)
{
    static std::vector<fi_cq_entry> entries;
    entries.resize(max);

    ssize_t ret = fi_cq_read(this->cq, entries.data(), max);
    if(ret > 0)
    {
        this->outstanding -= static_cast<int>(ret);
        return static_cast<int>(ret);
    }
    if(ret == -FI_EAGAIN)
    {
        return 0;
    }
    if(ret == -FI_EAVAIL)
    {
        fi_cq_err_entry err{};
        fi_cq_readerr(this->cq, &err, 0);
        ThrowOFIError("CQ error: " + std::string(fi_cq_strerror(this->cq, err.prov_errno, err.err_data, nullptr, 0)) +
                      " (err=" + std::to_string(err.err) + ", prov_errno=" + std::to_string(err.prov_errno) + ")");
    }
    ThrowOFIError("fi_cq_read failed", static_cast<int>(-ret));
    return 0;
}

void OFIContext::DrainCompletions()
{
    while(this->outstanding > 0)
    {
        this->PollCompletions(this->outstanding);
        if(this->outstanding > 0)
        {
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        }
    }
}

void OFIContext::MakeProgress()
{
    if(this->outstanding > 0)
    {
        this->PollCompletions(std::min(16, this->outstanding));
    }
    else
    {
        struct fi_cq_entry entry;
        ssize_t ret = fi_cq_read(this->cq, &entry, 1);
        if(ret > 0)
        {
            return;
        }
        if(ret < 0 && ret != -FI_EAGAIN)
        {
            if(ret == -FI_EAVAIL)
            {
                fi_cq_err_entry err{};
                fi_cq_readerr(this->cq, &err, 0);
                ThrowOFIError("CQ error during MakeProgress: " +
                    std::string(fi_cq_strerror(this->cq, err.prov_errno, err.err_data, nullptr, 0)) +
                    " (err=" + std::to_string(err.err) + ", prov_errno=" + std::to_string(err.prov_errno) + ")");
            }
            ThrowOFIError("fi_cq_read failed during MakeProgress", static_cast<int>(-ret));
        }
    }
}

void OFIContext::EnsureConnected(const std::vector<int> &peer_world_ranks, MPI_Comm /*exchange_comm*/)
{
    for(int wr : peer_world_ranks)
    {
        if(wr < 0 or wr >= this->size)
        {
            ThrowOFIError("EnsureConnected: peer world rank out of bounds");
        }
        if(this->connected_mode)
        {
            if(wr != this->rank and not this->peer_eps[wr])
            {
                ThrowOFIError("EnsureConnected: peer " + std::to_string(wr) +
                              " has no connected endpoint (MSG mode)");
            }
        }
        else
        {
            if(not this->peer_connected[wr])
            {
                ThrowOFIError("EnsureConnected: peer " + std::to_string(wr) +
                              " not in address vector");
            }
        }
    }
}

void OFIContext::Free()
{
    if(this->freed)
    {
        return;
    }

    this->DrainCompletions();

    for(fid_ep *peer_ep : this->peer_eps)
    {
        if(peer_ep)
        {
            fi_close(&peer_ep->fid);
        }
    }
    this->peer_eps.clear();

    if(this->ep)
    {
        fi_close(&this->ep->fid);
        this->ep = nullptr;
    }
    if(this->pep)
    {
        fi_close(&this->pep->fid);
        this->pep = nullptr;
    }
    if(this->av)
    {
        fi_close(&this->av->fid);
        this->av = nullptr;
    }
    if(this->eq)
    {
        fi_close(&this->eq->fid);
        this->eq = nullptr;
    }
    if(this->cq)
    {
        fi_close(&this->cq->fid);
        this->cq = nullptr;
    }
    if(this->domain)
    {
        fi_close(&this->domain->fid);
        this->domain = nullptr;
    }
    if(this->fabric)
    {
        fi_close(&this->fabric->fid);
        this->fabric = nullptr;
    }
    if(this->fi)
    {
        fi_freeinfo(this->fi);
        this->fi = nullptr;
    }

    this->freed = true;
}

#endif // __WITH_OFI
