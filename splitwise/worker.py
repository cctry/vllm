# from kv_comm import KVComm
import asyncio
import random
import time
from collections import defaultdict
from functools import lru_cache
from itertools import accumulate
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import utils
from rdma_transport import (
    TensorBlock,
    TensorBlocks,
    VllmRdmaClient,
    VllmRdmaServer,
)

from vllm.worker.worker import Worker

KV_TIMEOUT = 30


@lru_cache()
def to_byte(s):
    return s.encode("ascii")


def coalesce_indices(local_indices, remote_indices):
    sorted_indices = np.argsort(local_indices)
    local_indices = local_indices[sorted_indices]
    remote_indices = remote_indices[sorted_indices]
    local_diff = np.diff(local_indices)
    remote_diff = np.diff(remote_indices)
    breaks = np.where((local_diff != 1) | (remote_diff != 1))[0] + 1
    segment_starts = np.concatenate(([0], breaks))
    segment_ends = np.concatenate((breaks, [len(local_indices)]))
    local_start = local_indices[segment_starts]
    remote_start = remote_indices[segment_starts]
    num_blocks = segment_ends - segment_starts
    return local_start, remote_start, num_blocks


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
        self.kv_stride = cache[0].stride()[0]

        self.tensor_blocks = TensorBlocks()
        for c in cache:
            self.tensor_blocks.add(
                TensorBlock(c.data_ptr(), 0, c.numel() * c.element_size())
            )
        self.role = role
        if role == "server":
            self.server = VllmRdmaServer(
                self.local_addr, self.device_id, self.tensor_blocks
            )
            self.server.listen()
        elif role == "client":
            self.clients = {}
            self.remote_base_ptr = {}
        self.pending_requests = {}
        self.remaining_blocks = {}

    def get_client(self, server_addr):
        if server_addr not in self.clients:
            client = VllmRdmaClient(
                self.local_addr, self.device_id, self.tensor_blocks
            )
            remote_tb: TensorBlocks = client.connect(server_addr)
            self.clients[server_addr] = client
            remote_base_ptr = remote_tb.get_base_ptrs()
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
                local_start, remote_start, num_blocks = coalesce_indices(
                    np.array(local_bid), np.array(remote_bid)
                )
                for l_start, r_start, num_blk in zip(local_start, remote_start, num_blocks):
                    for layer in layers:
                        remote_base_ptr = remote_base[layer]
                        local_base_ptr = self.local_base[layer]
                        remaining -= num_blk
                        client.send(
                            TensorBlock(
                                local_base_ptr,
                                l_start * self.block_size,
                                self.block_size * num_blk
                            ),
                            TensorBlock(
                                remote_base_ptr,
                                r_start * self.block_size,
                                self.block_size * num_blk
                            ),
                            to_byte(rid),
                            remaining
                        )
                        remaining -= num_blk
                        client.send(
                            TensorBlock(
                                local_base_ptr,
                                self.kv_stride + l_start * self.block_size,
                                self.block_size * num_blk
                            ),
                            TensorBlock(
                                remote_base_ptr,
                                self.kv_stride + r_start * self.block_size,
                                self.block_size * num_blk
                            ),
                            to_byte(rid),
                            remaining
                        )
                self.remaining_blocks[rid] = remaining

    def wait_kv(self, request_id):
        if self.role == "server":
            handle = self.server
        elif self.role == "client":
            addr, _ = self.pending_requests[request_id]
            handle = self.clients[addr]
        start = time.time()
        while handle.is_complete(to_byte(request_id)) is None:
            if time.time() - start > KV_TIMEOUT:
                raise TimeoutError(f"KV cache pull for {request_id} timeout")
            time.sleep(1 / 1000)


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
                indices = list(accumulate(args[3].seq_lens))[:-1]
                # This ensures the computation of this layer finish
                block_ids = (slot_mapping // block_size).cpu()
                block_ids = block_ids.tensor_split(indices)
                block_ids = [
                    torch.unique_consecutive(bid).tolist() for bid in block_ids
                ]
                self.kv_comm.push_kv(args[3].request_ids, block_ids, layers)

                return output

            return new_forward if is_push else old_forward

        for i, layer in enumerate(layers):
            layer.forward = forward_wrapper(i, layer)

        return {"device": self.local_rank, "host": host}

    def wait_kv(self, request_id: str):
        """Wait for kv comm to finish."""
        self.kv_comm.wait_kv(request_id)  # block here

    def add_request(self, request_id, kv_addr, block_ids):
        info = next(
            info for info in kv_addr if info["device"] == self.local_rank
        )
        self.kv_comm.add_request(
            request_id, f"{info['host']}:{info['port']}", block_ids
        )
