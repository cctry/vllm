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
        return cache_shape, cache.dtype, cache.device

    def get_kv_blocks_data(
        self, block_ids: List[int]
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """This function returns a list of block metadata for CUDA IPC."""
        k_blocks = [
            utils.wrap_tensor(cache[0, block_id, ...])
            for block_id in block_ids
            for cache in self.gpu_cache
        ]
        v_blocks = [
            utils.wrap_tensor(cache[1, block_id, ...])
            for block_id in block_ids
            for cache in self.gpu_cache
        ]
        blocks = k_blocks + v_blocks
        tensor_data = [mp.reductions.reduce_tensor(block) for block in blocks]
        return tensor_data

    def decode_kv_init(self):
        """Initialize the KV cache communicator as the decode worker"""
        shape, dtype, device = self.setup()
        self._manager = mp.Manager()
        self.recv_flags = self._manager.dict()
        self.requests_queue = mp.Queue()
        self.kv_comm = KVComm(
            device,
            dtype,
            shape,
            "client",
            self.requests_queue,
            recv_flags=self.recv_flags,
        )
        self.kv_comm.start()

    def prefill_kv_init(self, port: int):
        """Initialize the KV cache communicator as the prefill worker"""
        shape, dtype, device = self.setup()
        self.requests_queue = mp.Queue()
        self.port = port + self.local_rank
        self.kv_comm = KVComm(
            device,
            dtype,
            shape,
            "server",
            self.requests_queue,
            server_port=self.port,
        )
        self.kv_comm.start()
        host = utils.set_NIC(self.local_rank, False)
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
        self.recv_flags[request_id] = False
        self.requests_queue.put(
            (
                request_id,
                info["host"],
                info["port"],
                tensor_data,
            )
        )
        start = time.time()
        while not self.recv_flags[request_id]:
            if time.time() - start > KV_TIMEOUT:
                raise TimeoutError("KV cache pull timeout")
            time.sleep(0.1)

    def push_kv(self, request_id: str, block_ids: List[int]):
        """Push the kv cache to the decode worker
        This function is non-blocking.
        """
        tensor_data = self.get_kv_blocks_data(block_ids)
        self.requests_queue.put((request_id, tensor_data))
