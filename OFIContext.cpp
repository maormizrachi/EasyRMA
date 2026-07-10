#include "OFIContext.hpp"

#ifdef __WITH_OFI

#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstdio>
#include <sys/uio.h>

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
      outstanding(0), freed(false)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);

    this->SetupFabric(provider_name);
    this->ExchangeAddresses();
}

OFIContext::~OFIContext()
{
    if(not std::uncaught_exceptions() and not this->freed)
    {
        this->Free();
    }
}

static fi_info *ChooseBestProvider(fi_info *list)
{
    fi_info *best = nullptr;
    int best_score = -1;

    for(fi_info *cur = list; cur != nullptr; cur = cur->next)
    {
        std::string pname = cur->fabric_attr->prov_name ? cur->fabric_attr->prov_name : "";
        bool virt_addr = (cur->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0;

        bool is_rxd = pname.find("ofi_rxd") != std::string::npos;
        bool is_shm = (pname == "shm");

        if(is_rxd or is_shm) continue;

        int score = 0;
        if(pname.find("cxi") != std::string::npos) score = 100;
        else if(pname.find("efa") != std::string::npos) score = 90;
        else if(pname.find("ofi_rxm") != std::string::npos) score = 80;
        else if(pname.find("tcp") != std::string::npos) score = 50;
        else score = 30;

        if(virt_addr) score += 10;

        if(score > best_score)
        {
            best_score = score;
            best = cur;
        }
    }

    return best;
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

void OFIContext::SetupFabric(const std::string &provider_name)
{
    fi_info *hints = fi_allocinfo();
    if(not hints)
    {
        ThrowOFIError("fi_allocinfo failed");
    }

    hints->caps = FI_RMA | FI_ATOMIC;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->domain_attr->threading = FI_THREAD_SAFE;
    hints->domain_attr->control_progress = FI_PROGRESS_AUTO;
    hints->domain_attr->data_progress = FI_PROGRESS_AUTO;

    if(not provider_name.empty())
    {
        hints->fabric_attr->prov_name = strdup(provider_name.c_str());
    }

    fi_info *info_list = nullptr;
    int ret = fi_getinfo(FI_VERSION(1, 6), nullptr, nullptr, 0, hints, &info_list);
    if(ret != 0)
    {
        hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED;
        ret = fi_getinfo(FI_VERSION(1, 6), nullptr, nullptr, 0, hints, &info_list);
    }
    if(ret != 0)
    {
        hints->domain_attr->mr_mode = 0;
        ret = fi_getinfo(FI_VERSION(1, 6), nullptr, nullptr, 0, hints, &info_list);
    }
    fi_freeinfo(hints);

    if(ret != 0)
    {
        ThrowOFIError("fi_getinfo failed — no suitable provider found", -ret);
    }

    fi_info *chosen = ChooseBestProvider(info_list);
    if(not chosen)
    {
        chosen = info_list;
    }

    this->fi = fi_dupinfo(chosen);
    fi_freeinfo(info_list);

    if(not this->fi)
    {
        ThrowOFIError("fi_dupinfo failed");
    }

    this->mr_mode = this->fi->domain_attr->mr_mode;
    this->mr_endpoint = (this->mr_mode & FI_MR_ENDPOINT) != 0;
    this->inject_size = this->fi->tx_attr->inject_size;

    if(this->rank == 0)
    {
        std::string provider = this->fi->fabric_attr->prov_name ?
            this->fi->fabric_attr->prov_name : "unknown";
        std::string component = OFIProviderComponent(provider);
        fprintf(stderr, "[OFI] component: %s (provider: %s)\n",
                component.c_str(), provider.c_str());
    }

    ret = fi_fabric(this->fi->fabric_attr, &this->fabric, nullptr);
    if(ret != 0)
    {
        ThrowOFIError("fi_fabric failed", -ret);
    }

    ret = fi_domain(this->fabric, this->fi, &this->domain, nullptr);
    if(ret != 0)
    {
        ThrowOFIError("fi_domain failed", -ret);
    }

    fi_cq_attr cq_attr{};
    cq_attr.size = this->cq_size;
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    ret = fi_cq_open(this->domain, &cq_attr, &this->cq, nullptr);
    if(ret != 0)
    {
        ThrowOFIError("fi_cq_open failed", -ret);
    }

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

void OFIContext::ExchangeAddresses()
{
    size_t addrlen = 0;
    fi_getname(&this->ep->fid, nullptr, &addrlen);

    std::vector<char> local_addr(addrlen);
    int ret = fi_getname(&this->ep->fid, local_addr.data(), &addrlen);
    if(ret != 0)
    {
        ThrowOFIError("fi_getname failed", -ret);
    }

    std::vector<char> all_addrs(this->size * addrlen);
    MPI_Allgather(local_addr.data(), static_cast<int>(addrlen), MPI_BYTE,
                  all_addrs.data(), static_cast<int>(addrlen), MPI_BYTE, this->comm);

    this->peer_addrs.resize(this->size);
    int inserted = fi_av_insert(this->av, all_addrs.data(), this->size,
                                this->peer_addrs.data(), 0, nullptr);
    if(inserted != this->size)
    {
        ThrowOFIError("fi_av_insert failed: inserted " + std::to_string(inserted) +
                      " of " + std::to_string(this->size) + " addresses");
    }

    this->peer_connected.assign(this->size, true);
}

fi_addr_t OFIContext::GetPeerAddr(int world_rank) const
{
    return this->peer_addrs[world_rank];
}

fid_mr *OFIContext::RegisterMemory(void *buf, size_t bytes, uint64_t access)
{
    fid_mr *mr = nullptr;

    uint64_t requested_key = 0;
    if(not (this->mr_mode & FI_MR_PROV_KEY))
    {
        requested_key = this->mr_key_counter++;
    }

    int ret = fi_mr_reg(this->domain, buf, bytes, access,
                        0, requested_key, 0, &mr, nullptr);
    if(ret != 0)
    {
        ThrowOFIError("fi_mr_reg failed", -ret);
    }

    if(this->mr_endpoint)
    {
        ret = fi_mr_bind(mr, &this->ep->fid, 0);
        if(ret != 0)
        {
            fi_close(&mr->fid);
            ThrowOFIError("fi_mr_bind failed", -ret);
        }
        ret = fi_mr_enable(mr);
        if(ret != 0)
        {
            fi_close(&mr->fid);
            ThrowOFIError("fi_mr_enable failed", -ret);
        }
    }

    return mr;
}

void OFIContext::DeregisterMemory(fid_mr *mr)
{
    if(mr)
    {
        fi_close(&mr->fid);
    }
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

    while(bytes > MAX_CHUNK)
    {
        this->EnsureCQSpace();
        ssize_t ret;
        do
        {
            ret = PostWriteWithCompletion(this->ep, this->peer_addrs[target_rank],
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
        ret = PostWriteWithCompletion(this->ep, this->peer_addrs[target_rank],
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
    ssize_t ret;
    do
    {
        ret = fi_inject_write(this->ep, local_addr, bytes,
                              this->peer_addrs[target_rank], remote_addr, rkey);
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

    while(bytes > MAX_CHUNK)
    {
        this->EnsureCQSpace();
        ssize_t ret;
        do
        {
            ret = PostReadWithCompletion(this->ep, this->peer_addrs[target_rank],
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
        ret = PostReadWithCompletion(this->ep, this->peer_addrs[target_rank],
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

void OFIContext::PostAtomicCAS(int target_rank,
                               const void *swap_val, void *swap_desc,
                               const void *compare_val, void *compare_desc,
                               void *result, void *result_desc,
                               uint64_t remote_addr, uint64_t rkey,
                               enum fi_datatype dtype, bool signaled)
{
    (void)signaled;
    this->EnsureCQSpace();

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
    msg.addr = this->peer_addrs[target_rank];
    msg.rma_iov = &rma_iov;
    msg.rma_iov_count = 1;
    msg.datatype = dtype;
    msg.op = FI_CSWAP;

    ssize_t ret;
    do
    {
        ret = fi_compare_atomicmsg(this->ep, &msg,
                                    &compare_iov, compare_descs, 1,
                                    &result_iov, result_descs, 1,
                                    FI_COMPLETION);
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
    msg.addr = this->peer_addrs[target_rank];
    msg.rma_iov = &rma_iov;
    msg.rma_iov_count = 1;
    msg.datatype = dtype;
    msg.op = FI_SUM;

    ssize_t ret;
    do
    {
        ret = fi_fetch_atomicmsg(this->ep, &msg,
                                  &result_iov, result_descs, 1,
                                  FI_COMPLETION);
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

void OFIContext::EnsureConnected(const std::vector<int> &peer_world_ranks, MPI_Comm /*exchange_comm*/)
{
    for(int wr : peer_world_ranks)
    {
        if(wr < 0 or wr >= this->size)
        {
            ThrowOFIError("EnsureConnected: peer world rank out of bounds");
        }
        if(not this->peer_connected[wr])
        {
            ThrowOFIError("EnsureConnected: peer " + std::to_string(wr) + " not in address vector");
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

    if(this->ep)
    {
        fi_close(&this->ep->fid);
        this->ep = nullptr;
    }
    if(this->av)
    {
        fi_close(&this->av->fid);
        this->av = nullptr;
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
