import time
from itertools import accumulate
from typing import Dict, List, Optional, Tuple

import block_copy
import torch
import torch.multiprocessing as mp
import utils
from kv_comm import KVComm
import random

from vllm.worker.worker import Worker

KV_TIMEOUT = 120
BUFFER_SIZE = 4 * 1024 * 1024  # 4MB


class WorkerSplitwise(Worker):
    def setup(self):
        mp.set_start_method("spawn")
        cache = self.gpu_cache[0]
        assert self.local_rank == cache.device.index
        block_shape = tuple(cache.shape)[-3:]
        self._manager = mp.Manager()
        self.flags = self._manager.dict()
        self.requests_queue = mp.Queue(1024)
        host = utils.set_NIC(self.local_rank, False)
        self.kv_addr = {}
        # compile the gather/scatter kernels
        if self.is_driver_worker:
            block_size = (
                cache.shape[-1]
                * cache.shape[-2]
                * cache.shape[-3]
                * cache.dtype.itemsize
            )
            num_packing_blocks = BUFFER_SIZE // block_size
            assert BUFFER_SIZE % block_size == 0
            block_copy.get_block_copy(num_packing_blocks, block_size)
        return block_shape, cache.dtype, cache.device, host

    def get_kv_blocks_data(
        self,
        block_ids: List[int],
        layers: Optional[List[int]] = None,
    ):
        """This function returns a list of block metadata for CUDA IPC.
        The blocks are ordered as tokens -> layers -> (k, v).
        For example, the layout of blocks is (S/16, L, 2 : 2L, 2, 1)
        """
        blocks = []
        if layers is None:
            layers = range(len(self.gpu_cache))
        for block_id in block_ids:
            for layer in (self.gpu_cache[i] for i in layers):
                blocks.append(layer[0, block_id, ...])
                blocks.append(layer[1, block_id, ...])
        blocks = [utils.wrap_tensor(block) for block in blocks]
        tensor_data = [mp.reductions.reduce_tensor(block) for block in blocks]
        return tensor_data

    def decode_kv_init(self, port: int):
        """Initialize the KV cache communicator as the decode worker"""
        shape, dtype, device, host = self.setup()
        self.port = port + self.local_rank
        addr = f"{host}:{self.port}"
        self.kv_comm = KVComm(
            device,
            dtype,
            shape,
            "server",
            self.requests_queue,
            self.flags,
            BUFFER_SIZE,
            local_addr=addr,
        )
        self.kv_comm.start()
        return {"device": self.local_rank, "host": host, "port": self.port}

    def prefill_kv_init(self, layer_wise=-1):
        """Initialize the KV cache communicator as the prefill worker"""
        shape, dtype, device, host = self.setup()
        self.kv_comm = KVComm(
            device,
            dtype,
            shape,
            "client",
            self.requests_queue,
            self.flags,
            BUFFER_SIZE,
            local_addr=f"{host}"
        )
        self.kv_comm.start()

        self.rust_th = start(KV_ptr)

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
                    self.push_kv(req_id, block_id.tolist(), layers, addr)
                return output

            return new_forward if is_push else old_forward

        for i, layer in enumerate(layers):
            layer.forward = forward_wrapper(i, layer)

        return {"device": self.local_rank, "host": host}

    def pull_kv(self, request_id: str, block_ids: List[int]):
        """Listen for the kv cache."""
        tensor_data = self.get_kv_blocks_data(block_ids)
        self.flags[request_id] = False
        self.rust_th.queue.put(
            (
                request_id,
                tensor_data,
            )
        )
        # start = time.time()
        # while not self.flags[request_id]:
        #     if time.time() - start > KV_TIMEOUT:
        #         raise TimeoutError(f"KV cache pull for {request_id} timeout")
        #     time.sleep(0.1)
        # self.flags.pop(request_id)

    def wait_pull_kv(self, request_id: str):
        start = time.time()
        while not self.flags[request_id]:
            if time.time() - start > KV_TIMEOUT:
                raise TimeoutError(f"KV cache pull for {request_id} timeout")
            time.sleep(1/1000)
        self.flags.pop(request_id)


    def push_kv(
        self, request_id: str, block_ids: List[int], layers: List[int], kv_addr
    ):
        """Push the kv cache to the decode worker"""
        info = next(
            info for info in kv_addr if info["device"] == self.local_rank
        )
        tensor_data = self.get_kv_blocks_data(block_ids, layers)
        self.requests_queue.put(
            (request_id, info["host"], info["port"], tensor_data)
        )

    def finish_push_kv(self, request_id: str):
        """Wait for the push_kv to finish.
        This function is non-blocking.
        """
        self.flags[request_id] = False
        self.requests_queue.put((request_id, None, None, None))
        start = time.time()
        while not self.flags[request_id]:
            if time.time() - start > KV_TIMEOUT:
                raise TimeoutError(f"KV cache push for {request_id} timeout")
            time.sleep(0.1)
        self.flags.pop(request_id)
        self.kv_addr.pop(request_id)

    def add_kv_addr(self, request_id, addr):
        assert request_id not in self.kv_addr
        self.kv_addr[request_id] = addr
