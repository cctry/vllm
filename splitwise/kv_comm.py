import asyncio

import block_copy
import torch
import torch.multiprocessing as mp
import ucp
import utils
import os
import struct


def serialize_int_list(int_list):
    byte_array = bytearray(len(int_list) * 4)
    for i, number in enumerate(int_list):
        struct.pack_into("i", byte_array, i * 4, number)
    return byte_array


def deserialize_bytearray(byte_array):
    int_list = [
        struct.unpack_from("i", byte_array, i * 4)[0]
        for i in range(len(byte_array) // 4)
    ]
    return int_list


def get_blk_seq(start, num, length):
    lst = list(range(start, start + num)) + [-1] * (length - num)
    return serialize_int_list(lst)

def from_blk_seq(blk_seq):
    lst = deserialize_bytearray(blk_seq)
    return [x for x in lst if x != -1]


class KVComm(mp.Process):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        shape: tuple,
        role: str,
        requests_queue: mp.Queue,
        flags: dict,
        buffer_size: int,
        server_port=None,
    ):
        super().__init__(name=f"KVComm:{device.index}")
        self.role = role
        self.device = device
        self.dtype = dtype
        self.requests_queue = requests_queue
        self.server_port = server_port
        self.flags = flags
        assert role == "client" or server_port is not None
        assert role == "server" or flags is not None
        self.block_shape = shape
        self.block_size = shape[0] * shape[1] * shape[2] * dtype.itemsize
        self.buffer_size = buffer_size
        self.num_packing_blocks = self.buffer_size // self.block_size
        assert self.buffer_size % self.block_size == 0
        assert self.num_packing_blocks != 0

    async def _kv_server_handler(self, ep: ucp.Endpoint):
        id_buffer = await ep.am_recv()
        request_id = id_buffer.decode("utf-8")
        while request_id not in self.pending_requests:
            await asyncio.sleep(0)
        tensor_data = self.pending_requests.pop(request_id)
        blocks = [func(*args) for (func, args) in tensor_data]
        remaining = len(blocks)
        buffer = self.get_buffer() # This buffer may from Rust
        while remaining > 0:
            blk_seq = await ep.am_recv() # Notified with the sent blocks
            await ep.recv(buffer) # Buffer is written
            idx = from_blk_seq(blk_seq)
            blks = [blocks[i] for i in idx] # Figure out the addresses
            self.block_copy.scatter(blks, buffer) # Scatter
            remaining -= len(blks)
        self.flags[request_id] = True
        await ep.close()

    def kv_server(self):
        """The server will listen for KV blocks"""

        async def _kv_server():
            self.pending_requests = {}
            self.lf = ucp.create_listener(
                self._kv_server_handler, self.server_port
            )
            while not self.lf.closed():
                if not self.requests_queue.empty():
                    request_id, tensor_data = self.requests_queue.get()
                    self.pending_requests[request_id] = tensor_data
                await asyncio.sleep(0.1)

        try:
            asyncio.run(_kv_server())
        except Exception as e:
            raise e
        finally:
            self.lf.close()

    async def _kv_push(self, host, port, request_id, tensor_queue):
        ep = await ucp.create_endpoint(host, port)
        id_buffer = bytearray(request_id, "utf-8")
        await ep.am_send(id_buffer)
        count = 0
        while True:
            tensor_data = await tensor_queue.get()
            if tensor_data is None:
                break
            buffer = self.get_buffer()
            blocks = [func(*args) for (func, args) in tensor_data]
            for blks in utils.chunk(blocks, self.num_packing_blocks):
                blk_seq = get_blk_seq(count, len(blks), self.num_packing_blocks)
                self.block_copy.gather(buffer, blks) # Gather blocks to buffer
                await ep.am_send(blk_seq) # Notify server what are written
                await ep.send(buffer) # Write remote buffer
                count += len(blks)
        self.flags[request_id] = True
        await ep.close()

    def kv_client(self):
        """The server will send KV blocks"""

        async def _kv_client():
            tasks = {}
            while True:
                if not self.requests_queue.empty():
                    request_id, host, port, tensor_data = (
                        self.requests_queue.get(True)
                    )
                    if request_id not in tasks:
                        queue = asyncio.Queue()
                        task = asyncio.create_task(
                            self._kv_push(host, port, request_id, queue),
                            name=request_id,
                        )
                        tasks[request_id] = (task, queue)
                        task.add_done_callback(
                            lambda task: tasks.pop(task.get_name())
                        )
                    tasks[request_id][1].put_nowait(tensor_data)
                await asyncio.sleep(0.1)

        asyncio.run(_kv_client())

    def get_buffer(self) -> torch.Tensor:
        # TODO: We can use double buffer here
        buffer = torch.empty(
            (self.num_packing_blocks, *self.block_shape),
            device=self.device,
            dtype=self.dtype,
        )
        return utils.wrap_tensor(buffer)

    def run(self):
        torch.cuda.init()
        self.block_copy = block_copy.get_block_copy(
            self.num_packing_blocks, self.block_size
        )
        utils.set_NIC(self.device.index)
        method_map = {"server": self.kv_server, "client": self.kv_client}
        method_map[self.role]()
