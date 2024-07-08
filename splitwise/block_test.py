import asyncio
import time
import ucp
import torch
import utils
import argparse


port = 13337


def mc_mb_client():
    async def recv(data):
        ep = await ucp.create_endpoint(args.host, port)
        for i in range(args.nblk):
            await ep.recv(data[i])
        await ep.close()
        
        
    async def main():
        start = time.time()
        await asyncio.gather(*[recv(data[i]) for i in range(args.nreq)])
        bandwidth = total_size / (time.time() - start) / 2**20
        return bandwidth
    return asyncio.run(main())


def sc_mb_client():
    async def recv():
        ep = await ucp.create_endpoint(args.host, port)
        for r in range(args.nreq):
            for i in range(args.nblk):
                await ep.recv(data[r, i, ...])
        await ep.close()

    start = time.time()
    asyncio.run(recv())
    bandwidth = total_size / (time.time() - start) / 2**20
    return bandwidth
    

def sc_sb_client():
    async def recv():
        ep = await ucp.create_endpoint(args.host, port)
        for i in range(args.nreq):
            await ep.recv(data[i])
        await ep.close()
    start = time.time()
    asyncio.run(recv())
    bandwidth = total_size / (time.time() - start) / 2**20
    return bandwidth
    
    
    
def mc_sb_client():
    async def recv(data):
        ep = await ucp.create_endpoint(args.host, port)
        await ep.recv(data)
        await ep.close()
        
    async def main():
        start = time.time()
        await asyncio.gather(*[recv(data[i]) for i in range(args.nreq)])
        return total_size / (time.time() - start) / 2**20
    return asyncio.run(main())
    

def mc_mb_server():
    async def send(ep):
        for i in range(args.nblk):
            await ep.send(data[0,i,...])
        await ep.close()

    async def main():
        try:
            global lf
            lf = ucp.create_listener(send, port)
            while not lf.closed():
                await asyncio.sleep(0.1)
        except Exception as e:
            print(e)
            lf.close()

    asyncio.run(main())
    lf.close()


def sc_mb_server():
    async def send(ep):
        for r in range(args.nreq):
            for i in range(args.nblk):
                await ep.send(data[r, i, ...])
        await ep.close()

    async def main():
        try:
            global lf
            lf = ucp.create_listener(send, port)
            while not lf.closed():
                await asyncio.sleep(0.1)
        except Exception as e:
            print(e)
            lf.close()

    asyncio.run(main())
    lf.close()
    
def sc_sb_server():
    async def send(ep):
        for i in range(args.nreq):
            await ep.send(data[i])
        await ep.close()

    async def main():
        try:
            global lf
            lf = ucp.create_listener(send, port)
            while not lf.closed():
                await asyncio.sleep(0.1)
        except Exception as e:
            print(e)
            lf.close()

    asyncio.run(main())
    lf.close()
    
def mc_sb_server():
    async def send(ep):
        await ep.send(data[0])
        await ep.close()

    async def main():
        try:
            global lf
            lf = ucp.create_listener(send, port)
            while not lf.closed():
                await asyncio.sleep(0.1)
        except Exception as e:
            print(e)
            lf.close()

    asyncio.run(main())
    lf.close()


if __name__ == "__main__":
    host = utils.set_NIC(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--nblk", type=int, required=True)
    parser.add_argument("--blksize", type=int, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--nreq", type=int, default=10)
    parser.add_argument("--method", type=str, default="sc_mb")
    args = parser.parse_args()
    
    blk_size = args.blksize * 1024
    total_size = args.nreq * blk_size * args.nblk
    
    data = torch.empty(
        (args.nreq, args.nblk, blk_size), dtype=torch.uint8, device="cuda"
    )
    if args.role == "server":
        print(f"Listening on {host}:{port}")
    
    func = globals()[f"{args.method}_{args.role}"]
    func()
    
    if args.role == "client":
        avg_bandwidth = sum(func() for _ in range(5)) / 5
        print(f"Average bandwidth: {avg_bandwidth} MB/s")
    

