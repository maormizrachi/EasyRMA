#include "IBVContext.hpp"

#ifdef __WITH_IBV

#include <cstring>
#include <cassert>
#include <algorithm>

static void ThrowIBVError(const std::string &context, int err = 0)
{
    std::string msg = "IBVContext: " + context;
    if(err != 0)
    {
        msg += " (errno=" + std::to_string(err) + ": " + strerror(err) + ")";
    }
    throw std::runtime_error(msg);
}

IBVContext::IBVContext(MPI_Comm comm, const std::string &device_name, uint8_t ib_port, int gid_index, int cq_size)
    : comm(comm), ctx(nullptr), pd(nullptr), cq(nullptr),
      ib_port(ib_port), gid_index(gid_index),
      outstanding(0), cq_size(cq_size), max_inline_data(0), freed(false), is_roce(false)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);

    this->ctx = this->OpenDevice(device_name);

    ibv_port_attr port_attr{};
    if(ibv_query_port(this->ctx, this->ib_port, &port_attr) != 0)
    {
        ThrowIBVError("ibv_query_port failed", errno);
    }

    this->lid = port_attr.lid;
    this->is_roce = (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET);

    if(this->is_roce or this->lid == 0)
    {
        if(ibv_query_gid(this->ctx, this->ib_port, this->gid_index, &this->gid) != 0)
        {
            ThrowIBVError("ibv_query_gid failed", errno);
        }
    }
    else
    {
        std::memset(&this->gid, 0, sizeof(this->gid));
    }

    this->pd = ibv_alloc_pd(this->ctx);
    if(not this->pd)
    {
        ThrowIBVError("ibv_alloc_pd failed", errno);
    }

    this->cq = ibv_create_cq(this->ctx, this->cq_size, nullptr, nullptr, 0);
    if(not this->cq)
    {
        ThrowIBVError("ibv_create_cq failed", errno);
    }

    this->qps.assign(this->size, nullptr);
    this->qp_connected.assign(this->size, false);
    this->outstanding_per_target.assign(this->size, 0);
}

IBVContext::~IBVContext()
{
    if(not std::uncaught_exceptions() and not this->freed)
    {
        this->Free();
    }
}

ibv_context *IBVContext::OpenDevice(const std::string &device_name)
{
    int num_devices = 0;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if(not dev_list or num_devices == 0)
    {
        ThrowIBVError("no InfiniBand devices found");
    }

    ibv_device *target = nullptr;
    if(device_name.empty())
    {
        target = dev_list[0];
    }
    else
    {
        for(int i = 0; i < num_devices; i++)
        {
            if(device_name == ibv_get_device_name(dev_list[i]))
            {
                target = dev_list[i];
                break;
            }
        }
    }

    if(not target)
    {
        ibv_free_device_list(dev_list);
        ThrowIBVError("device '" + device_name + "' not found");
    }

    ibv_context *ctx = ibv_open_device(target);
    ibv_free_device_list(dev_list);

    if(not ctx)
    {
        ThrowIBVError("ibv_open_device failed", errno);
    }

    return ctx;
}

ibv_qp *IBVContext::CreateSingleQP()
{
    ibv_qp_init_attr init_attr{};
    init_attr.send_cq = this->cq;
    init_attr.recv_cq = this->cq;
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.cap.max_send_wr = DEFAULT_MAX_SEND_WR;
    init_attr.cap.max_recv_wr = 1;
    init_attr.cap.max_send_sge = DEFAULT_MAX_SGE;
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_inline_data = 256;
    init_attr.sq_sig_all = 0;

    ibv_qp *qp = ibv_create_qp(this->pd, &init_attr);
    if(not qp)
    {
        ThrowIBVError("ibv_create_qp failed", errno);
    }

    if(this->max_inline_data == 0)
    {
        this->max_inline_data = init_attr.cap.max_inline_data;
    }

    return qp;
}

void IBVContext::EnsureConnected(const std::vector<int> &peer_world_ranks, MPI_Comm exchange_comm)
{
    int exchange_size;
    MPI_Comm_size(exchange_comm, &exchange_size);
    assert(static_cast<int>(peer_world_ranks.size()) == exchange_size);

    bool any_new = false;
    for(int i = 0; i < exchange_size; i++)
    {
        int wr = peer_world_ranks[i];
        if(wr < 0 or wr >= this->size)
        {
            ThrowIBVError("EnsureConnected: peer world rank out of bounds");
        }
        if(this->qps[wr] == nullptr or not this->qp_connected[wr])
        {
            any_new = true;
            break;
        }
    }

    int global_need = 0;
    int local_need = any_new ? 1 : 0;
    MPI_Allreduce(&local_need, &global_need, 1, MPI_INT, MPI_MAX, exchange_comm);
    if(global_need == 0)
        return;

    std::vector<IBVConnectionInfo> send_buf(exchange_size, IBVConnectionInfo{});
    std::vector<IBVConnectionInfo> recv_buf(exchange_size);

    for(int i = 0; i < exchange_size; i++)
    {
        int wr = peer_world_ranks[i];
        if(wr < 0 or wr >= this->size)
        {
            ThrowIBVError("EnsureConnected: peer world rank out of bounds");
        }
        if(this->qps[wr] == nullptr)
        {
            ibv_qp *qp = this->CreateSingleQP();
            this->TransitionToInit(qp);
            this->qps[wr] = qp;
            this->qpn_to_rank[qp->qp_num] = wr;
        }

        send_buf[i].lid = this->lid;
        send_buf[i].gid = this->gid;
        send_buf[i].psn = 0;
        send_buf[i].qpn = this->qps[wr]->qp_num;
    }

    MPI_Alltoall(send_buf.data(), static_cast<int>(sizeof(IBVConnectionInfo)), MPI_BYTE,
                 recv_buf.data(), static_cast<int>(sizeof(IBVConnectionInfo)), MPI_BYTE,
                 exchange_comm);

    for(int i = 0; i < exchange_size; i++)
    {
        int wr = peer_world_ranks[i];
        if(this->qp_connected[wr])
        {
            continue;
        }
        if(recv_buf[i].qpn == 0)
        {
            ThrowIBVError("EnsureConnected: missing remote QP info");
        }

        if(wr == this->rank)
        {
            IBVConnectionInfo self_info{};
            self_info.lid = this->lid;
            self_info.gid = this->gid;
            self_info.psn = 0;
            self_info.qpn = this->qps[wr]->qp_num;
            this->TransitionToRTR(this->qps[wr], self_info, 0);
        }
        else
        {
            this->TransitionToRTR(this->qps[wr], recv_buf[i], 0);
        }
        this->TransitionToRTS(this->qps[wr], 0);
        this->qp_connected[wr] = true;
    }
}

void IBVContext::TransitionToInit(ibv_qp *qp)
{
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = this->ib_port;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if(ibv_modify_qp(qp, &attr, flags) != 0)
    {
        ThrowIBVError("transition to INIT failed", errno);
    }
}

void IBVContext::TransitionToRTR(ibv_qp *qp, const IBVConnectionInfo &remote, uint32_t /*local_psn*/)
{
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote.qpn;
    attr.rq_psn = remote.psn;
    attr.max_dest_rd_atomic = 16;
    attr.min_rnr_timer = 12;

    attr.ah_attr.port_num = this->ib_port;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;

    if(this->is_roce)
    {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = remote.gid;
        attr.ah_attr.grh.sgid_index = this->gid_index;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.traffic_class = 0;
    }
    else
    {
        attr.ah_attr.dlid = remote.lid;
        attr.ah_attr.is_global = 0;
    }

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    if(ibv_modify_qp(qp, &attr, flags) != 0)
    {
        ThrowIBVError("transition to RTR failed", errno);
    }
}

void IBVContext::TransitionToRTS(ibv_qp *qp, uint32_t local_psn)
{
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = local_psn;
    attr.max_rd_atomic = 16;

    int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

    if(ibv_modify_qp(qp, &attr, flags) != 0)
    {
        ThrowIBVError("transition to RTS failed", errno);
    }
}

void IBVContext::PostSend(ibv_qp *qp, ibv_send_wr &wr)
{
    ibv_send_wr *bad_wr = nullptr;
    if(ibv_post_send(qp, &wr, &bad_wr) != 0)
    {
        ThrowIBVError("ibv_post_send failed", errno);
    }

    if(wr.send_flags & IBV_SEND_SIGNALED)
    {
        this->outstanding++;
        auto it = this->qpn_to_rank.find(qp->qp_num);
        if(it == this->qpn_to_rank.end())
        {
            ThrowIBVError("unknown QP completion source");
        }
        this->outstanding_per_target[it->second]++;
    }
}

void IBVContext::EnsureCQSpace()
{
    int threshold = std::min(this->cq_size / 2, DEFAULT_MAX_SEND_WR - 2);
    if(this->outstanding >= threshold)
    {
        this->DrainCompletions();
    }
}

void IBVContext::PostRDMAWrite(int target_rank, const void *local_addr, size_t bytes, uint32_t lkey, uint64_t remote_addr, uint32_t rkey, bool signaled)
{
    this->EnsureCQSpace();

    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(local_addr);
    sge.length = static_cast<uint32_t>(bytes);
    sge.lkey = lkey;

    ibv_send_wr wr{};
    wr.wr_id = 0;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    if(bytes <= this->max_inline_data)
    {
        wr.send_flags |= IBV_SEND_INLINE;
    }
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;

    this->PostSend(this->qps[target_rank], wr);
}

void IBVContext::PostRDMAWriteBatch(int target_rank, const RDMAWriteEntry *entries, size_t count, uint32_t lkey, uint32_t rkey)
{
    static std::vector<ibv_sge> sges;
    static std::vector<ibv_send_wr> wrs;

    if(count == 0)
    {
        return;
    }

    ibv_qp *qp = this->qps[target_rank];
    int target_for_tracking = -1;
    {
        auto it = this->qpn_to_rank.find(qp->qp_num);
        if(it == this->qpn_to_rank.end())
        {
            ThrowIBVError("PostRDMAWriteBatch: unknown QP");
        }
        target_for_tracking = it->second;
    }

    size_t max_batch = static_cast<size_t>(std::max(1, std::min(this->cq_size / 2, DEFAULT_MAX_SEND_WR - 2)));

    size_t pos = 0;
    while(pos < count)
    {
        this->DrainCompletions();

        size_t batch_size = std::min(count - pos, max_batch);
        sges.resize(batch_size);
        wrs.resize(batch_size);

        for(size_t i = 0; i < batch_size; i++)
        {
            const RDMAWriteEntry &e = entries[pos + i];

            sges[i] = {};
            sges[i].addr = reinterpret_cast<uint64_t>(e.local_addr);
            sges[i].length = e.bytes;
            sges[i].lkey = lkey;

            wrs[i] = {};
            wrs[i].wr_id = 0;
            wrs[i].sg_list = &sges[i];
            wrs[i].num_sge = 1;
            wrs[i].opcode = IBV_WR_RDMA_WRITE;
            wrs[i].send_flags = (e.bytes <= this->max_inline_data) ? IBV_SEND_INLINE : 0;
            wrs[i].wr.rdma.remote_addr = e.remote_addr;
            wrs[i].wr.rdma.rkey = rkey;
            wrs[i].next = (i + 1 < batch_size) ? &wrs[i + 1] : nullptr;
        }

        wrs[batch_size - 1].send_flags |= IBV_SEND_SIGNALED;

        ibv_send_wr *bad_wr = nullptr;
        if(ibv_post_send(qp, &wrs[0], &bad_wr) != 0)
        {
            ThrowIBVError("PostRDMAWriteBatch: ibv_post_send failed", errno);
        }

        this->outstanding++;
        this->outstanding_per_target[target_for_tracking]++;

        pos += batch_size;
    }
}

void IBVContext::PostRDMARead(int target_rank, void *local_addr, size_t bytes, uint32_t lkey, uint64_t remote_addr, uint32_t rkey, bool signaled)
{
    this->EnsureCQSpace();

    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(local_addr);
    sge.length = static_cast<uint32_t>(bytes);
    sge.lkey = lkey;

    ibv_send_wr wr{};
    wr.wr_id = 0;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;

    this->PostSend(this->qps[target_rank], wr);
}

void IBVContext::PostAtomicCAS(int target_rank, uint64_t *local_addr, uint32_t lkey, uint64_t remote_addr, uint32_t rkey, uint64_t compare_val, uint64_t swap_val, bool signaled)
{
    this->EnsureCQSpace();

    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(local_addr);
    sge.length = 8;
    sge.lkey = lkey;

    ibv_send_wr wr{};
    wr.wr_id = 0;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    wr.wr.atomic.remote_addr = remote_addr;
    wr.wr.atomic.rkey = rkey;
    wr.wr.atomic.compare_add = compare_val;
    wr.wr.atomic.swap = swap_val;

    this->PostSend(this->qps[target_rank], wr);
}

void IBVContext::PostAtomicFetchAdd(int target_rank, uint64_t *local_addr, uint32_t lkey, uint64_t remote_addr, uint32_t rkey, uint64_t add_val, bool signaled)
{
    this->EnsureCQSpace();

    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(local_addr);
    sge.length = 8;
    sge.lkey = lkey;

    ibv_send_wr wr{};
    wr.wr_id = 0;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
    wr.wr.atomic.remote_addr = remote_addr;
    wr.wr.atomic.rkey = rkey;
    wr.wr.atomic.compare_add = add_val;

    this->PostSend(this->qps[target_rank], wr);
}

int IBVContext::PollCompletions(int max)
{
    static std::vector<ibv_wc> wc;
    wc.resize(max);
    int n = ibv_poll_cq(this->cq, max, wc.data());
    if(n < 0)
    {
        ThrowIBVError("ibv_poll_cq failed");
    }

    for(int i = 0; i < n; i++)
    {
        if(wc[i].status != IBV_WC_SUCCESS)
        {
            ThrowIBVError("work completion error: " + std::string(ibv_wc_status_str(wc[i].status)) + " (vendor_err=" + std::to_string(wc[i].vendor_err) + ")");
        }

        auto it = this->qpn_to_rank.find(wc[i].qp_num);
        if(it == this->qpn_to_rank.end())
        {
            ThrowIBVError("completion received for unknown QP");
        }
        int target_rank = it->second;
        if(target_rank < 0 or target_rank >= this->size)
        {
            ThrowIBVError("completion target rank out of bounds");
        }
        if(this->outstanding_per_target[target_rank] > 0)
        {
            this->outstanding_per_target[target_rank]--;
        }
    }
    this->outstanding -= n;
    return n;
}

void IBVContext::DrainCompletions()
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

void IBVContext::DrainCompletionsForTarget(int target_rank)
{
    if(target_rank < 0 or target_rank >= this->size)
    {
        ThrowIBVError("DrainCompletionsForTarget: target rank out of bounds");
    }

    while(this->outstanding_per_target[target_rank] > 0)
    {
        int batch = std::max(1, std::min(this->outstanding, this->cq_size / 2));
        this->PollCompletions(batch);
    }
}

void IBVContext::Free()
{
    if(this->freed)
    {
        return;
    }

    this->DrainCompletions();

    for(ibv_qp *qp : this->qps)
    {
        if(qp)
        {
            ibv_destroy_qp(qp);
        }
    }
    this->qps.clear();
    this->qp_connected.clear();

    if(this->cq)
    {
        ibv_destroy_cq(this->cq);
        this->cq = nullptr;
    }
    if(this->pd)
    {
        ibv_dealloc_pd(this->pd);
        this->pd = nullptr;
    }
    if(this->ctx)
    {
        ibv_close_device(this->ctx);
        this->ctx = nullptr;
    }

    this->freed = true;
}

#endif // __WITH_IBV
