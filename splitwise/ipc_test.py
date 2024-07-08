import ucp
import torch
import asyncio
import torch.multiprocessing as mp
import utils

block_shape = (16, 8, 128)

class Server(mp.Process):
    def __init__(self, device_id, port, id_queue, obj_queue):
        super().__init__()
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.port = port
        self.id_queue = id_queue
        self.loop = asyncio.new_event_loop()
        self.obj_queue = obj_queue
    
    async def handler(self, ep):
        blocks = [0, 1, 2, 3]
        self.id_queue.put(blocks)
        k_meta, v_meta = self.obj_queue.get()
        k_blocks = [k[0](*k[1]) for k in k_meta]
        v_blocks = [v[0](*v[1]) for v in v_meta]
        
        blocks = k_blocks + v_blocks
        # self.buffer[0].copy_(blocks[0], non_blocking=True)
        # for i, block in enumerate(blocks[1:]):
        #     self.buffer[i%2].copy_(block, non_blocking=True)
        #     await ep.send(self.buffer[(i+1)%2])
        # ep.send(self.buffer[len(blocks) % 2])
        for block in blocks:
            await ep.recv(self.buffer)
            block.copy_(self.buffer)
            
        await ep.close()
        self.lf.close()
            
    async def main(self):
        self.lf = ucp.create_listener(self.handler, self.port)
        while not self.lf.closed():
            await asyncio.sleep(0.1)

    def run(self):
        # self.buffer = torch.empty((2, *block_shape), device=self.device)
        self.buffer = torch.empty(block_shape, device=self.device)
        asyncio.set_event_loop(self.loop)
        host = utils.set_NIC(self.device_id)
        print(f"Server on {host}")
        self.loop.run_until_complete(self.main())
        self.id_queue.put(None)


class Client(mp.Process):
    def __init__(self, device_id, host, port, id_queue, obj_queue):
        super().__init__()
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.loop = asyncio.new_event_loop()
        self.host = host
        self.port = port
        self.id_queue = id_queue
        self.obj_queue = obj_queue
    
    async def main(self):
        blocks = [0, 1, 20, 30]
        self.id_queue.put(blocks)
        ep = await ucp.create_endpoint(self.host, self.port)
        k_meta, v_meta = self.obj_queue.get()
        k_blocks = [k[0](*k[1]) for k in k_meta]
        v_blocks = [v[0](*v[1]) for v in v_meta]
        
        blocks = k_blocks + v_blocks
        # self.buffer[0].copy_(blocks[0], non_blocking=True)
        # for i, block in enumerate(blocks[1:]):
        #     self.buffer[(i+1)%2].copy_(block, non_blocking=True)
        #     await ep.send(self.buffer[i%2])
        # ep.send(self.buffer[len(blocks) % 2])
        for block in blocks:
            self.buffer.copy_(block)
            await ep.send(self.buffer)
        
       
        print(f"Sent {len(blocks)} blocks")
        
        await ep.close()
        for block in k_blocks + v_blocks:
            block.mul_(2)
        
    def run(self):
        # self.buffer = torch.empty((2, *block_shape), device=self.device)
        # self.cuda_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self.buffer = torch.empty(block_shape, device=self.device)
        asyncio.set_event_loop(self.loop)
        utils.set_NIC(self.device_id)
        self.loop.run_until_complete(self.main())
        self.id_queue.put(None)
        


class KVCacheHolder(mp.Process):
    def __init__(self, device_id, id_queue, obj_queue, value):
        super().__init__()
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.id_queue = id_queue
        self.obj_queue = obj_queue
        self.value = value

    def run(self):
        self.kv_cache = torch.full((2, 134, *block_shape), self.value, device=self.device)
        block_ids = None
        while True:
            cmd = self.id_queue.get()
            if cmd is None:
                break
            block_ids = cmd
            k_blocks = [
                self.kv_cache[0, block_id, ...] for block_id in block_ids
            ]
            v_blocks = [
                self.kv_cache[1, block_id, ...] for block_id in block_ids
            ]
            print(f"KV cache holder k_address {[hex(blk.data_ptr()) for blk in k_blocks]}")
            print(f"KV cache holder v_address {[hex(blk.data_ptr()) for blk in v_blocks]}")
            k_blocks_meta = [mp.reductions.reduce_tensor(k) for k in k_blocks]
            v_blocks_meta = [mp.reductions.reduce_tensor(v) for v in v_blocks]
            self.obj_queue.put((k_blocks_meta, v_blocks_meta))
        print(f"KV cache {block_ids} done")
        print(f"Modified {self.kv_cache[:, block_ids, ...]}")


import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ucx-test.py <role> <value>")
        sys.exit(1)
    role = sys.argv[1]
    value = float(sys.argv[2])

    device_id = 0
    
    kv_cache_id_queue = mp.Queue()
    kv_cache_obj_queue = mp.Queue()
    kv_cache = KVCacheHolder(device_id, kv_cache_id_queue, kv_cache_obj_queue, value)
    kv_cache.start()
    if role == "server":
        server = Server(device_id, 13337, kv_cache_id_queue, kv_cache_obj_queue)
        server.start()
    else:
        host = sys.argv[3]
        client = Client(device_id, host, 13337, kv_cache_id_queue, kv_cache_obj_queue)
        client.start()
    kv_cache.join()
    print("Done")