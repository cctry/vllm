import asyncio
import concurrent
import os
import threading
from typing import Dict, List, Tuple

import torch
import ucp
import utils

from vllm.worker.worker import Worker
from kv_comm import KVComm
import torch.multiprocessing as mp

KV_TIMEOUT = 30


class WorkerSplitwise(Worker):
    def get_kv_blocks_data(
        self, block_ids: List[int]
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """This function returns a list of block metadata for CUDA IPC."""
        k_blocks = [
            cache[0, block_id, ...]
            for block_id in block_ids
            for cache in self.gpu_cache
        ]
        v_blocks = [
            cache[1, block_id, ...]
            for block_id in block_ids
            for cache in self.gpu_cache
        ]
        blocks = k_blocks + v_blocks
        # torch's array interface does not support bfloat16.
        # We may need similar fix for other types like fp8
        if blocks[0].dtype == torch.bfloat16:
            blocks = [block.view(torch.float16) for block in blocks]

        tensor_data = [mp.reductions.reduce_tensor(block) for block in blocks]
        return tensor_data

    def decode_kv_init(self):
        """Initialize the KV cache communicator as the decode worker"""
        cache = self.gpu_cache[0]
        assert self.local_rank == cache.device.index
        self.requests_queue = mp.Queue()
        self.kv_comm = KVComm(
            cache.device,
            cache.dtype,
            cache.shape[2:],
            "client",
            self.requests_queue,
        )
        self.kv_comm.start()

    def prefill_kv_init(self, port: int):
        """Initialize the KV cache communicator as the prefill worker"""
        cache = self.gpu_cache[0]
        assert self.local_rank == cache.device.index
        self.requests_queue = mp.Queue()
        self.port = port + self.local_rank
        self.kv_comm = KVComm(
            cache.device,
            cache.dtype,
            cache.shape[2:],
            "server",
            self.requests_queue,
            self.port,
        )
        self.kv_comm.start()
        host = utils.set_NIC(self.local_rank)
        return {"device": self.local_rank, "host": host, "port": self.port}

    def pull_kv(
        self, request_id: str, block_ids: List[int], kv_server_info: Dict
    ):
        """Wait for the kv cache to be pulled.
        This function is blocking and will wait until the kv cache is pulled.
        """
        info = next(
            info for info in kv_server_info if info["device"] == self.local_rank
        )
        tensor_data = self.get_kv_blocks_data(block_ids)        
        event = mp.Event()
        self.requests_queue.put((
            request_id,
            info["host"],
            info["port"],
            tensor_data,
            event,
        ))
        event.wait(KV_TIMEOUT)

    def push_kv(self, request_id: str, block_ids: List[int]):
        """Push the kv cache to the decode worker
        This function is non-blocking.
        """
        tensor_data = self.get_kv_blocks_data(block_ids)
        self.requests_queue.put((request_id, tensor_data))
        