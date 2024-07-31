# from kv_comm import KVComm
import random
import time
from collections import defaultdict
from itertools import accumulate
from typing import Dict, List, Optional, Tuple

import rdma_transport
import torch
import torch.multiprocessing as mp
import utils
import asyncio
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
                self.local_addr,
                self.device_id,
            )
            self.server.listen()
        elif role == "client":
            self.clients = {}
            self.remote_base_ptr = {}
        self.pending_requests = {}
        self.remaining_blocks = {}

        self.loop = asyncio.get_event_loop()

    def get_client(self, server_addr):
        if server_addr not in self.clients:
            client = rdma_transport.RdmaClient(
                self.local_addr,
                self.device_id,
                # base_ptr?
            )
            remote_base_ptr = client.connect(server_addr)  # Can we do this?
            self.clients[server_addr] = client
            self.remote_base_ptr[server_addr] = remote_base_ptr
        else:
            client = self.clients[server_addr]
            remote_base_ptr = self.remote_base_ptr[server_addr]
        return client, remote_base_ptr

    def add_request(
        self, request_id: str, server_addr: str, block_ids: List[int]
    ):
        assert request_id not in self.pending_requests
        self.pending_requests[request_id] = (server_addr, block_ids)
        self.remaining_blocks[request_id] = len(block_ids) * self.num_layer * 2

    def push_kv(
        self,
        request_ids: List[str],
        local_block_ids: List[List[int]],
        layers: List[int],
    ):
        requests = defaultdict(list)
        for rid, local_bid in zip(request_ids, local_block_ids):
            addr, remote_bid = self.pending_requests[rid]
            requests[addr].append((rid, local_bid, remote_bid))

        for addr, request_info in requests.items():
            client, remote_base = self.get_client(addr)
            for rid, local_bid, remote_bid in request_info:
                remaining = self.remaining_blocks[rid]
                for layer in layers:
                    remote_base_ptr = remote_base[layer]
                    local_base_ptr = self.local_base[layer]
                    for l_bid, r_bid in zip(local_bid, remote_bid):
                        client.send(
                            local_base_ptr,
                            remote_base_ptr,
                            l_bid * self.block_size,  # local offset
                            r_bid * self.block_size,  # remote offset
                            self.block_size,  # size
                            rid,  # request id
                            remaining - 1,  # remaining
                        )
                        client.send(
                            local_base_ptr,
                            remote_base_ptr,
                            self.kv_stride
                            + l_bid * self.block_size,  # local offset
                            self.kv_stride
                            + r_bid * self.block_size,  # remote offset
                            self.block_size,  # size
                            rid,  # request id
                            remaining - 2,  # remaining
                        )
                        remaining -= 2
                self.remaining_blocks[rid] = remaining

        def wait_kv(self, request_id):
            # TODO: Wait for the request to finish
            self.pending_requests.pop(request_id)


class WorkerSplitwise(Worker):
    def setup(self):
        # Assume all tensors are the same
        cache = self.gpu_cache[0]
        assert self.local_rank == cache.device.index
        info = utils.detect_NIC(self.local_rank)
        addr = info["address"]
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
                # TODO: If there is chunked prefill, we need to get the actual computed block ids for this iteration.
                # Besides, we need to figure out the remote block id for this iteration as well.
                # For now, the chunked prefill has not benefits, so we just push the whole layer.
                slot_mapping = args[3].slot_mapping
                block_ids = (slot_mapping // block_size).cpu()
                indices = list(accumulate(args[3].seq_lens))[:-1]
                block_ids = block_ids.tensor_split(indices)
                block_ids = [
                    torch.unique_consecutive(bid).tolist() for bid in block_ids
                ]
                # CSY: Also, we might want to get the local block ids from sequence metadata 
                # to avoid the D2H transfer blocking CPU

                self.kv_comm.push_kv(args[3].request_ids, block_ids, layers)

                return output

            return new_forward if is_push else old_forward

        for i, layer in enumerate(layers):
            layer.forward = forward_wrapper(i, layer)

        return {"device": self.local_rank, "host": host}

    def finish_push_kv(self, request_id: str):
        """Wait for the push_kv to finish."""
        self.kv_comm.wait_kv(request_id)  # block here

    def add_request(
        self, request_id, kv_addr, block_ids
    ):  # Will be called from driver
        assert request_id not in self.pending_requests
        info = next(
            info for info in kv_addr if info["device"] == self.local_rank
        )
        self.kv_comm.add_request(
            request_id, f"{info['host']}:{info['port']}", block_ids
        )
