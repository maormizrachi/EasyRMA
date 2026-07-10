#ifndef OFI_CONTEXT_HPP
#define OFI_CONTEXT_HPP

#ifdef __WITH_OFI

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <mpi.h>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

struct OFIRemoteRegion
{
    uint64_t addr;
    uint64_t key;
};

class OFIContext
{
public:
    static constexpr int DEFAULT_CQ_SIZE = 4096;

    OFIContext(MPI_Comm comm, const std::string &provider_name = "");

    ~OFIContext();

    fid_domain *GetDomain() const { return this->domain; }
    fid_ep *GetEP() const { return this->ep; }
    fid_cq *GetCQ() const { return this->cq; }
    MPI_Comm GetComm() const { return this->comm; }
    int GetRank() const { return this->rank; }
    int GetSize() const { return this->size; }
    size_t GetInjectSize() const { return this->inject_size; }
    bool NeedsMREndpoint() const { return this->mr_endpoint; }
    int GetMRMode() const { return this->mr_mode; }
    bool UsesVirtAddr() const { return (this->mr_mode & FI_MR_VIRT_ADDR) != 0; }

    fi_addr_t GetPeerAddr(int world_rank) const;

    fid_mr *RegisterMemory(void *buf, size_t bytes, uint64_t access);
    void DeregisterMemory(fid_mr *mr);

    void PostRDMAWrite(int target_rank, const void *local_addr, size_t bytes,
                       void *desc, uint64_t remote_addr, uint64_t rkey, bool signaled = true);

    void PostInjectWrite(int target_rank, const void *local_addr, size_t bytes,
                         uint64_t remote_addr, uint64_t rkey);

    void PostRDMARead(int target_rank, void *local_addr, size_t bytes,
                      void *desc, uint64_t remote_addr, uint64_t rkey, bool signaled = true);

    void PostAtomicCAS(int target_rank,
                       const void *swap_val, void *swap_desc,
                       const void *compare_val, void *compare_desc,
                       void *result, void *result_desc,
                       uint64_t remote_addr, uint64_t rkey,
                       enum fi_datatype dtype, bool signaled = true);

    void PostAtomicFetchAdd(int target_rank,
                            const void *addend, void *addend_desc,
                            void *result, void *result_desc,
                            uint64_t remote_addr, uint64_t rkey,
                            enum fi_datatype dtype, bool signaled = true);

    int PollCompletions(int max = 1);
    void DrainCompletions();

    void EnsureConnected(const std::vector<int> &peer_world_ranks, MPI_Comm exchange_comm);

    void Free();

private:
    MPI_Comm comm;
    int rank, size;
    fi_info *fi;
    fid_fabric *fabric;
    fid_domain *domain;
    fid_cq *cq;
    fid_av *av;
    fid_ep *ep;
    size_t inject_size;
    int mr_mode;
    bool mr_endpoint;
    uint64_t mr_key_counter;
    int cq_size;
    int outstanding;
    bool freed;

    std::vector<fi_addr_t> peer_addrs;
    std::vector<bool> peer_connected;

    void SetupFabric(const std::string &provider_name);
    void ExchangeAddresses();
    void EnsureCQSpace();
};

#endif // __WITH_OFI

#endif // OFI_CONTEXT_HPP
