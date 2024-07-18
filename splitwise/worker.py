from typing import Dict, List, Tuple

import torch
import time
import utils

from vllm.worker.worker import Worker
from kv_comm import KVComm
import torch.multiprocessing as mp

KV_TIMEOUT = 30


class WorkerSplitwise(Worker):
    def setup(self):
        mp.set_start_method("spawn")
        cache = self.gpu_cache[0]
        assert self.local_rank == cache.device.index
        cache_shape = (len(self.gpu_cache),) + tuple(cache.shape)
        self._manager = mp.Manager()
        self.flags = self._manager.dict()
        self.requests_queue = mp.Queue()
        host = utils.set_NIC(self.local_rank, False)
        return cache_shape, cache.dtype, cache.device, host

    def get_kv_blocks_data(
        self, block_ids: List[int]
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """This function returns a list of block metadata for CUDA IPC.
        The blocks are ordered as tokens -> layers -> (k, v).
        For example, the layout of blocks is (S/16, L, 2 : 2L, 2, 1)
        """
        blocks = []
        for block_id in block_ids:
            for layer in self.gpu_cache:
                blocks.append(layer[0, block_id, ...])
                blocks.append(layer[1, block_id, ...])
        tensor_data = [mp.reductions.reduce_tensor(block) for block in blocks]
        return tensor_data

    def decode_kv_init(self, port: int):
        """Initialize the KV cache communicator as the decode worker"""
        shape, dtype, device, host = self.setup()
        self.port = port + self.local_rank
        self.kv_comm = KVComm(
            device,
            dtype,
            shape,
            "server",
            self.requests_queue,
            self.flags,
            server_port=self.port,
        )
        self.kv_comm.start()
        return {"device": self.local_rank, "host": host, "port": self.port}

    def prefill_kv_init(self):
        """Initialize the KV cache communicator as the prefill worker"""
        shape, dtype, device, host = self.setup()
        self.kv_comm = KVComm(
            device, dtype, shape, "client", self.requests_queue, self.flags
        )
        self.kv_comm.start()
        return {"device": self.local_rank, "host": host}

    def pull_kv(self, request_id: str, block_ids: List[int]):
        """Listen for the kv cache."""
        tensor_data = self.get_kv_blocks_data(block_ids)
        self.flags[request_id] = False
        self.requests_queue.put(
            (
                request_id,
                tensor_data,
            )
        )
        start = time.time()
        while not self.flags[request_id]:
            if time.time() - start > KV_TIMEOUT:
                raise TimeoutError(f"KV cache pull for {request_id} timeout")
            time.sleep(0.1)
        self.flags.pop(request_id)

    def push_kv(self, request_id: str, block_ids: List[int], kv_addr):
        """Push the kv cache to the decode worker"""
        tensor_data = self.get_kv_blocks_data(block_ids)
        self.flags[request_id] = False
        self.requests_queue.put(
            (request_id, kv_addr["host"], kv_addr["port"], tensor_data)
        )
        start = time.time()
        while request_id not in self.flags:
            if time.time() - start > KV_TIMEOUT:
                raise TimeoutError(f"KV cache push for {request_id} timeout")
            time.sleep(0.1)
        self.flags.pop(request_id)

    def finish_push_kv(self, request_id: str):
        """Wait for the push_kv to finish.
        This function is non-blocking.
        """
        self.requests_queue.put(
            (request_id, None, None, None)
        )

