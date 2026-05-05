#ifndef IBV_CONTEXT_HPP
#define IBV_CONTEXT_HPP

#ifdef __WITH_IBV

#include <infiniband/verbs.h>
#include <mpi.h>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

struct IBVConnectionInfo
{
    uint16_t lid;
    uint32_t qpn;
    uint32_t psn;
    ibv_gid gid;
};

struct IBVRemoteRegion
{
    uint64_t addr;
    uint32_t rkey;
};

class IBVContext
{
public:
    static constexpr int DEFAULT_CQ_SIZE = 4096;
    static constexpr int DEFAULT_MAX_SEND_WR = 128;
    static constexpr int DEFAULT_MAX_SGE = 1;

    IBVContext(MPI_Comm comm, const std::string &device_name = "", uint8_t ib_port = 1, int gid_index = 0, int cq_size = DEFAULT_CQ_SIZE);

    ~IBVContext();

    ibv_pd *GetPD() const { return this->pd; }
    ibv_cq *GetCQ() const { return this->cq; }
    ibv_qp *GetQP(int rank) const { return this->qps[rank]; }
    MPI_Comm GetComm() const { return this->comm; }
    int GetRank() const { return this->rank; }
    int GetSize() const { return this->size; }
    uint32_t GetMaxInlineData() const { return this->max_inline_data; }

    void PostRDMAWrite(int target_rank, const void *local_addr, size_t bytes, uint32_t lkey, uint64_t remote_addr, uint32_t rkey, bool signaled = true);

    struct RDMAWriteEntry
    {
        const void *local_addr;
        uint32_t bytes;
        uint64_t remote_addr;
    };

    void PostRDMAWriteBatch(int target_rank, const RDMAWriteEntry *entries, size_t count, uint32_t lkey, uint32_t rkey);

    void PostRDMARead(int target_rank, void *local_addr, size_t bytes, uint32_t lkey, uint64_t remote_addr, uint32_t rkey, bool signaled = true);

    void PostAtomicCAS(int target_rank, uint64_t *local_addr, uint32_t lkey, uint64_t remote_addr, uint32_t rkey, uint64_t compare_val, uint64_t swap_val, bool signaled = true);

    void PostAtomicFetchAdd(int target_rank, uint64_t *local_addr, uint32_t lkey, uint64_t remote_addr, uint32_t rkey, uint64_t add_val, bool signaled = true);

    int PollCompletions(int max = 1);
    void DrainCompletions();
    void DrainCompletionsForTarget(int target_rank);

    void EnsureConnected(const std::vector<int> &peer_world_ranks, MPI_Comm exchange_comm);

    void Free();

private:
    MPI_Comm comm;
    int rank, size;
    ibv_context *ctx;
    ibv_pd *pd;
    ibv_cq *cq;
    uint8_t ib_port;
    int gid_index;
    uint16_t lid;
    ibv_gid gid;
    bool is_roce;
    std::vector<ibv_qp*> qps;
    std::vector<bool> qp_connected;
    std::unordered_map<uint32_t, int> qpn_to_rank;
    std::vector<int> outstanding_per_target;
    int outstanding;
    int cq_size;
    uint32_t max_inline_data;
    bool freed;

    ibv_context *OpenDevice(const std::string &device_name);
    ibv_qp *CreateSingleQP();
    void TransitionToInit(ibv_qp *qp);
    void TransitionToRTR(ibv_qp *qp, const IBVConnectionInfo &remote, uint32_t local_psn);
    void TransitionToRTS(ibv_qp *qp, uint32_t local_psn);
    void PostSend(ibv_qp *qp, ibv_send_wr &wr);
    void EnsureCQSpace();
};

#endif // __WITH_IBV

#endif // IBV_CONTEXT_HPP
