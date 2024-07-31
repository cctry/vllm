import asyncio

import block_copy
import torch
import torch.multiprocessing as mp
import ucp
import utils
import os
import struct
import rdma_transport
import random
import datetime

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


def parse_metadata(data):
    request_id = data[:32].decode('utf-8')
    bid = from_blk_seq(data[32:])
    return request_id, bid

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
        local_addr=None,
    ):
        super().__init__(name=f"KVComm:{device.index}")
        self.role = role
        self.device = device
        self.dtype = dtype
        self.requests_queue = requests_queue
        self.local_addr = local_addr
        self.flags = flags
        assert role == "server" or flags is not None
        self.block_shape = shape
        self.block_size = shape[0] * shape[1] * shape[2] * dtype.itemsize
        self.buffer_size = buffer_size
        self.num_packing_blocks = self.buffer_size // self.block_size
        assert self.buffer_size % self.block_size == 0
        assert self.num_packing_blocks != 0
        if role == "client":
            self.clients = {}

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
            self.server = rdma_transport.RdmaServer(self.local_addr, self.device.index)
            self.server.listen()
            self.remaining_blks = {}
            self.blk_flags = {}
            while True:
                # identifier = random.randint(0, 65536)
                # print(f"[{datetime.datetime.now()}] wait {identifier}")
                msg = await self.server.recv_message()
                assert msg is not None
                buffer = msg.get_buffer()[0] + msg.get_buffer()[1] * self.buffer_size
                metadata = msg.get_data()
                request_id, block_id = parse_metadata(metadata)
                # print(f"[{datetime.datetime.now()}] receive {request_id} {block_id[0]} {identifier}")
                # print(f"Buffer {buffer} {request_id} Received blocks {block_id}")
                # while not self.requests_queue.empty():
                #     request_id, tensor_data = self.requests_queue.get()
                #     blocks = [func(*args) for (func, args) in tensor_data]
                #     self.pending_requests[request_id] = blocks
                #     self.remaining_blks[request_id] = len(blocks)
                counter = 0
                # while request_id not in self.pending_requests:
                while self.pending_requests.get(request_id, None) is None:
                    new_request_id, tensor_data = self.requests_queue.get(False)
                    # print(f"Request {new_request_id} wants {len(tensor_data)} blocks")
                    blocks = [func(*args) for (func, args) in tensor_data]
                    self.pending_requests[new_request_id] = blocks
                    self.remaining_blks[new_request_id] = len(blocks)
                    counter += 1
                    
                    # self.blk_flags[new_request_id] = [False] * len(blocks) 
                # print(f"[{datetime.datetime.now()}] while end")
                print(f"While times {counter} {request_id}")
                assert request_id in self.pending_requests, f"{request_id} {block_id}"
                blocks = self.pending_requests[request_id]
                blks = [blocks[i] for i in block_id] # Figure out the addresses
                # for i in block_id:
                #     # assert self.blk_flags[request_id][i] is False, f"Repeated metadata for {request_id} and block {i}"
                #     if self.blk_flags[request_id][i] is True:
                #         print(f"Repeated metadata for {request_id} and block {i}")
                #     self.blk_flags[request_id][i] = True

                self.block_copy.scatter_ptr(blks, buffer) # Scatter
                # print(f"[{datetime.datetime.now()}] scatter end")
                self.remaining_blks[request_id] -= len(blks)
                # print(f"[{datetime.datetime.now()}] Flag IPC start")
                if self.remaining_blks[request_id] == 0:
                    self.flags[request_id] = True
                # print(f"[{datetime.datetime.now()}] Flag IPC end")
                # assert self.remaining_blks[request_id] >= 0, f"{request_id} received too many blocks"

        try:
            asyncio.run(_kv_server())
        except Exception as e:
            raise e
        finally:
            pass

    async def _kv_push(self, host, port, request_id, tensor_queue, slot):
        # print(f"{request_id} using slot {slot}")
        server_addr = f"{host}:{port}"
        if server_addr not in self.clients:
            local_addr = f"{self.local_addr}:{random.randint(40000, 60000)}"
            client = rdma_transport.RdmaClient(local_addr, self.device.index)
            client.connect(server_addr)
            self.clients[server_addr] = client
        else:
            client = self.clients[server_addr]
        # local_addr = f"{self.local_addr}:{random.randint(40000, 60000)}"
        # client = rdma_transport.RdmaClient(local_addr, self.device.index)
        # client.connect(server_addr)
        count = 0
        while True:
            tensor_data = await tensor_queue.get()
            if tensor_data is None:
                break
            buffer = client.get_buffer()[0]
            blocks = [func(*args) for (func, args) in tensor_data]
            slots = list(range(slot[0], slot[1]))
            for r, blks in enumerate(utils.chunk(blocks, self.num_packing_blocks)):
                blk_seq = get_blk_seq(count, len(blks), self.num_packing_blocks)
                metadata = bytearray(request_id, "utf-8") + blk_seq
                # print(f"[{request_id}] {list(range(count, count + len(blks))) + [-1] * (self.num_packing_blocks - len(blks))}")
                s = slots[r % len(slots)]
                self.block_copy.gather_ptr(buffer + s * self.buffer_size, blks) # Gather blocks to buffer
                await client.send(s, self.buffer_size, metadata)
                count += len(blks)
        self.flags[request_id] = True
        # client.shutdown()

    def kv_client(self):
        """The server will send KV blocks"""

        async def _kv_client():
            tasks = {}
            self.rid = 0
            while True:
                if not self.requests_queue.empty():
                    request_id, host, port, tensor_data = (
                        self.requests_queue.get(True)
                    )
                    if request_id not in tasks:
                        rid = self.rid 
                        self.rid += 1
                        slot_base = (rid % (256 // 8)) * 8
                        slot = (slot_base, slot_base + 8)
                        queue = asyncio.Queue()
                        task = asyncio.create_task(
                            self._kv_push(host, port, request_id, queue, slot),
                            name=request_id,
                        )
                        tasks[request_id] = (task, queue)
                        task.add_done_callback(
                            lambda task: tasks.pop(task.get_name())
                        )
                    tasks[request_id][1].put_nowait(tensor_data)
                await asyncio.sleep(0.1)
        # TODO: try and shutdown
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
