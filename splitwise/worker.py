import asyncio
import concurrent
import os
import threading
from typing import Dict, List, Tuple

import torch
import ucp
import utils

from vllm.worker.worker import Worker

KV_TIMEOUT = 30

class WorkerSplitwise(Worker):
    def setup_kv_comm(self, kv_comm_func, *args, **kwargs):
        self.pending_requests: Dict[str, Tuple[List[int], asyncio.Event]] = {}
        self.lock = threading.Lock()
        self.loop = asyncio.new_event_loop()
        assert self.local_rank == self.gpu_cache[0].device.index
        host = utils.set_NIC(self.local_rank)
        self.kv_thread = threading.Thread(
            target=kv_comm_func,
            args=(self.loop, *args),
            kwargs=kwargs,
            daemon=True,
        )
        self.kv_thread.start()
        return host

    def get_kv_blocks(
        self, block_ids: List[int]
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """This function returns a list of blocks and their tags."""
        k_blocks = [
            cache[0, block_id, ...]
            for block_id in block_ids
            for cache in self.gpu_cache
        ]
        k_tags = [f"k:{i}" for i in range(len(k_blocks))]
        v_blocks = [
            cache[1, block_id, ...]
            for block_id in block_ids
            for cache in self.gpu_cache
        ]
        v_tags = [f"v:{i}" for i in range(len(v_blocks))]
        blocks  = k_blocks + v_blocks
        tags = k_tags + v_tags
        if self.gpu_cache[0].dtype == torch.bfloat16:
            blocks = [block.view(torch.float16) for block in blocks]
        return blocks, tags

    async def _kv_server_handler(self, ep: ucp.Endpoint):
        request_id = utils.get_empty_uuid_tensor("cuda")
        await ep.recv(request_id)
        request_id = utils.tensor_to_uuid(request_id)
        async with utils.async_lock(self.lock):
            block_ids, event = self.pending_requests.pop(request_id, None)
        blocks, tags = self.get_kv_blocks(block_ids)
        coros = [
            ep.send(block, utils.hash(tag), True)
            for block, tag in zip(blocks, tags)
        ]
        await asyncio.gather(*coros)
        event.set()  # Mark these blocks are sent
        await ep.close()

    async def _kv_server(self, port):
        self.lf = ucp.create_listener(self._kv_server_handler, port)
        while not self.lf.closed():
            await asyncio.sleep(1)

    async def _kv_pull(self, host, port, request_id, block_ids):
        # TODO: We can reuse this endpoint
        ep = await ucp.create_endpoint(host, port)
        id_tensor = utils.uuid_to_tensor(request_id, "cuda")
        await ep.send(id_tensor)
        blocks, tags = self.get_kv_blocks(block_ids)
        coros = [
            ep.recv(block, utils.hash(tag), True)
            for block, tag in zip(blocks, tags)
        ]
        await asyncio.gather(*coros)
        await ep.close()

    def pull_kv(
        self, request_id: str, block_ids: List[int], kv_server_info: Dict
    ):
        """Wait for the kv cache to be pulled.
        This function is blocking and will wait until the kv cache is pulled.
        """
        info = next(info for info in kv_server_info if info['device'] == self.local_rank)
        host = info['host']
        port = info['port']
        future = asyncio.run_coroutine_threadsafe(
            self._kv_pull(host, port, request_id, block_ids), self.loop
        )
        try:
            future.result(timeout=KV_TIMEOUT)
        except concurrent.futures.TimeoutError:
            print(f"Blocks of request {request_id} took too long to receive")
        except Exception as e:
            raise e

    def push_kv(self, request_id: str, block_ids: List[int]):
        """Push the kv cache to the decode worker
        This function is blocking until the blocks are sent.
        """
        with self.lock:  # Mark these blocks are ready
            event = asyncio.Event(loop=self.loop)
            self.pending_requests[request_id] = (block_ids, event)

        async def _coro():
            await event.wait()

        future = asyncio.run_coroutine_threadsafe(_coro(), self.loop)
        try:
            future.result(timeout=KV_TIMEOUT)
        except concurrent.futures.TimeoutError:
            print(f"Blocks of request {request_id} took too long to send")
        except Exception as e:
            raise e

    def decode_kv_init(self):
        """Initialize the KV cache communicator as the decode worker
        This function will set up a thread to run a async loop to receive the cache
        """

        def run(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self.setup_kv_comm(run)

    def prefill_kv_init(self, port: int):
        """Initialize the KV cache communicator as the prefill worker
        This function will set up the ucx listener for pull requests
        """
        self.port = port + self.local_rank

        def run(loop, port):
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._kv_server(port))
            except Exception as e:
                print(e)
                self.lf.close()

        host = self.setup_kv_comm(run, self.port)
        return {"device": self.local_rank, "host": host, "port": self.port}
