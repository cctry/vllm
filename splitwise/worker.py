import time
from itertools import accumulate
from typing import Dict, List, Optional, Tuple
import rdma_transport

import torch
import torch.multiprocessing as mp
import utils

# from kv_comm import KVComm
import random

from vllm.worker.worker import Worker


KV_TIMEOUT = 30
BUFFER_SIZE = 1 * 1024 * 1024  # 4MB


class KVComm:
    def __init__(
        self,
        cache: List[torch.Tensor],
        local_addr: str,
        role: str,
    ):
        self.cache = cache
        self.rank = cache[0].device.index
        self.device_id = utils.get_real_device_id(self.rank)
        self.local_addr = local_addr
        self.num_layer = len(cache)
        self.shape = tuple(cache[0].shape)
        self.block_size = (
            self.shape[-1]
            * self.shape[-2]
            * self.shape[-3]
            * cache[0].dtype.itemsize
        )  # in bytes
        self.local_base = [t.data_ptr() for t in cache]
        self.kv_stride = cache[0].stride[0]
        # TODO: start RDAM thread
        if role == "server":
            self.server = rdma_transport.RdmaServer(
                self.local_addr, self.device_id
            )
        elif role == "client":
            self.clients = {}
            self.remote_base_ptr = {}

    def push_kv(self, request_id, kv_addr, layers, block_ids):
        info = next(info for info in kv_addr if info["device"] == self.rank)
        server_addr = f'{info["host"]}:{info["port"]}'
        if server_addr not in self.clients:
            client = rdma_transport.RdmaClient(self.local_addr, self.device_id)
            remote_base_ptr = client.connect(server_addr)  # Can we do this?
            self.clients[server_addr] = client
            self.remote_base_ptr[server_addr] = remote_base_ptr
        else:
            client = self.clients[server_addr]
            remote_base_ptr = self.remote_base_ptr[server_addr]

        for blk in block_ids:
            for layer in layers:
                local_base = self.local_base[layer]
                remote_base = remote_base_ptr[layer]
                client.send(
                    local_base,
                    remote_base,
                    blk * self.block_size,
                    # metadata?
                )
                client.send(
                    local_base + self.kv_stride,
                    remote_base + self.kv_stride,
                    blk * self.block_size,
                    # metadata?
                )
        
        def wait_kv(self, request_id):
            pass # TODO


class WorkerSplitwise(Worker):
    def setup(self):
        # Assume all tensors are the same
        cache = self.gpu_cache[0]
        assert self.local_rank == cache.device.index
        info = utils.detect_NIC(self.local_rank)
        addr = info["address"]
        self.kv_addr = {}
        return addr

    def decode_kv_init(self, port: int):
        """Initialize the KV cache communicator as the decode worker"""
        host = self.setup()
        self.port = port + self.local_rank
        addr = f"{host}:{self.port}"
        self.kv_comm = KVComm(self.gpu_cache, addr, "server")
        return {"device": self.local_rank, "host": host, "port": self.port}

    def prefill_kv_init(self, layer_wise=-1):
        """Initialize the KV cache communicator as the prefill worker"""
        host = self.setup()
        port = random.randint(40000, 60000)
        self.kv_comm = KVComm(self.gpu_cache, f"{host}:{port}", "client")

        layers = self.model_runner.model.model.layers
        block_size = self.model_runner.block_size
        num_layer = len(self.gpu_cache)
        assert num_layer == len(layers)

        if layer_wise == -1:
            layer_wise = len(self.gpu_cache)

        assert layer_wise <= num_layer and num_layer % layer_wise == 0

        def forward_wrapper(i, layer):
            finished = i + 1
            is_push = finished % layer_wise == 0
            last_push = finished - layer_wise
            layers = list(range(last_push, finished))

            old_forward = layer.forward

            def new_forward(*args, **kwargs):
                output = old_forward(*args, **kwargs)
                slot_mapping = args[3].slot_mapping
                block_ids = (slot_mapping // block_size).cpu()
                indices = list(accumulate(args[3].seq_lens))[:-1]
                block_ids = block_ids.tensor_split(indices)
                for req_id, block_id in zip(args[3].request_ids, block_ids):
                    addr = self.kv_addr[req_id]
                    block_id = torch.unique_consecutive(block_id)
                    self.kv_comm.push_kv(
                        req_id, block_id.tolist(), layers, addr
                    )
                return output

            return new_forward if is_push else old_forward

        for i, layer in enumerate(layers):
            layer.forward = forward_wrapper(i, layer)

        return {"device": self.local_rank, "host": host}

    def finish_push_kv(self, request_id: str):
        """Wait for the push_kv to finish.
        """
        self.kv_comm.wait_kv(request_id) # block here
        self.kv_addr.pop(request_id)

    def add_kv_addr(self, request_id, addr): # Will be called from driver
        assert request_id not in self.kv_addr
        self.kv_addr[request_id] = addr
