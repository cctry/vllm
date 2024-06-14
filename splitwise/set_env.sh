conda install -c conda-forge -c rapidsai ucx-proc=*=gpu ucx<1.14 ucx-py
conda env config vars UCX_TLS=rc,sm,cuda_copy,cuda_ipc 
conda env config vars UCX_IB_GPU_DIRECT_RDMA=y