import asyncio

import torch
import torch.multiprocessing as mp
import ucp
import utils

import time
from contextlib import contextmanager
@contextmanager
def timer(request_id, job_desc):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[timer] {request_id}: {job_desc}: {elapsed}")

class KVComm(mp.Process):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        block_shape: tuple,
        role: str,
        requests_queue: mp.Queue,
        server_port=None,
        recv_flags=None,
    ):
        super().__init__()
        self.role = role
        self.device = device
        self.dtype = dtype
        self.block_shape = block_shape
        self.requests_queue = requests_queue
        self.server_port = server_port
        self.recv_flags = recv_flags
        assert role == "client" or server_port is not None
        assert role == "server" or recv_flags is not None

    async def _kv_server_handler(self, ep: ucp.Endpoint):
        id_tensor = utils.get_empty_uuid_tensor(self.device)
        buffer = self.get_buffer()
        await ep.recv(id_tensor)
        request_id = utils.tensor_to_uuid(id_tensor)
        with timer(request_id, "retrive ID"):
            while request_id not in self.pending_requests:
                await asyncio.sleep(0)
        tensor_data = self.pending_requests.pop(request_id)
        blocks = [data[0](*data[1]) for data in tensor_data]
        for block in blocks:
            buffer.copy_(block)
            await ep.send(buffer)
        with timer(request_id, "close"):
            await ep.close()   

    def kv_server(self):
        async def _kv_server():
            self.pending_requests = {}
            self.lf = ucp.create_listener(
                self._kv_server_handler, self.server_port
            )
            while not self.lf.closed():
                if not self.requests_queue.empty():
                    request_id, tensor_data = self.requests_queue.get()
                    self.pending_requests[request_id] = tensor_data
                await asyncio.sleep(0)

        asyncio.run(_kv_server())

    async def _kv_pull(self, host, port, request_id, tensor_data):
        with timer(request_id, "pull kv"):
            with timer(request_id, "connect"):
                ep = await ucp.create_endpoint(host, port)
            with timer(request_id, "exchange ID"):
                id_tensor = utils.uuid_to_tensor(request_id, self.device)
                await ep.send(id_tensor)
            buffer = self.get_buffer()
            blocks = [data[0](*data[1]) for data in tensor_data]
            for block in blocks:
                await ep.recv(buffer)
                block.copy_(buffer)
            self.recv_flags[request_id] = True
            with timer(request_id, "close"):
                await ep.close()    
    
    def kv_client(self):
        async def _kv_client():
            tasks = set()
            while True:
                if not self.requests_queue.empty():
                    request_id, host, port, tensor_data = (
                        self.requests_queue.get(True)
                    )
                    task = asyncio.create_task(
                        self._kv_pull(host, port, request_id, tensor_data)
                    )
                    tasks.add(task)
                    task.add_done_callback(tasks.remove)
                await asyncio.sleep(0)

        asyncio.run(_kv_client())

    def get_buffer(self) -> torch.Tensor:
        # TODO: We can use double buffer here
        buffer = torch.empty(
            self.block_shape, device=self.device, dtype=self.dtype
        )
        return utils.wrap_tensor(buffer)


    def run(self):
        torch.cuda.init()
        utils.set_NIC(self.device.index)
        method_map = {"server": self.kv_server, "client": self.kv_client}
        method_map[self.role]()
