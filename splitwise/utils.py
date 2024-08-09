import asyncio
import base64
import contextlib
import fcntl
import json
import os
import pickle
import socket
import struct
import subprocess
import time
import zlib
from contextlib import contextmanager
from functools import lru_cache, partial
from typing import Callable, Set

import torch
import ucp

from vllm.sequence import SequenceGroup
from vllm.utils import make_async


@lru_cache()
def to_byte(s):
    return s.encode("ascii")


class Recorder:
    def __init__(self, desc):
        self.desc = desc
        self.tagged_time = {}
        self.tagged_count = {}

    @contextmanager
    def record(self, tag):
        start = time.time()
        yield
        end = time.time()
        elapsed_time = end - start
        if tag not in self.tagged_time:
            self.tagged_time[tag] = elapsed_time
            self.tagged_count[tag] = 1
        else:
            self.tagged_time[tag] += elapsed_time
            self.tagged_count[tag] += 1

    def show(self):
        for tag, t in self.tagged_time.items():
            print(f"{self.desc}:{tag} {t}")


@contextmanager
def timer(desc):
    t = Recorder(desc)
    start = time.time()
    yield t
    end = time.time()
    print(f"{desc}: {end - start}")
    t.show()


def chunk(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def hash(*args):
    return zlib.adler32(repr(args).encode("utf-8"))


def deserialize_seq_group(data: str) -> SequenceGroup:
    data = base64.b64decode(data)
    return pickle.loads(data)


def serialize_seq_group(seq_group: SequenceGroup) -> str:
    data = pickle.dumps(seq_group)
    return base64.b64encode(data).decode("utf-8")


@contextlib.asynccontextmanager
async def async_lock(lock):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lock.acquire)
    try:
        yield  # the lock is held
    finally:
        lock.release()


def get_real_device_id(device_id):
    if device_id is None:
        device_id = torch.cuda.current_device()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if visible_devices:
        visible_devices = visible_devices.split(",")
        device_id = int(visible_devices[device_id])
    return device_id


def get_address(ifname):
    ifname = ifname.encode()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        return socket.inet_ntoa(
            fcntl.ioctl(
                s.fileno(),
                0x8915,
                struct.pack("256s", ifname[:15]),  # SIOCGIFADDR
            )[20:24]
        )


def dump_NIC():
    num_gpu = torch.cuda.device_count()
    topo_proc = subprocess.Popen(
        ["nvidia-smi", "topo", "-m"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rdma_proc = subprocess.Popen(
        ["rdma", "link"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    rdma_out = rdma_proc.communicate()[0]
    active_flag = "state ACTIVE physical_state LINK_UP"
    rdma_link = {}
    for link in rdma_out.split("\n"):
        if active_flag in link:
            mlx = link.split(" ")[1][:-2]
            eth = link.split(" ")[-2]
            rdma_link[mlx] = eth

    data = {}

    topo_out = topo_proc.communicate()[0]
    topo_lines = topo_out.split("\n")
    devices = topo_lines[0][5:].split()[:-4]
    for i in range(num_gpu):
        device = devices[i]
        conns = topo_lines[i + 1].split()[:-2]
        assert conns[0] == device, "Pasing error"
        assert len(conns[1:]) == len(devices), "Parsing error"
        mlxs = [devices[i] for i, conn in enumerate(conns[1:]) if conn == "PXB"]
        assert (
            len(mlxs) > 0
        ), f"Unable to find NIC connected to GPU {i} with PXB"
        nics = [
            (mlx, rdma_link[mlx])
            for mlx in mlxs
            if mlx in rdma_link and int(rdma_link[mlx][-1]) % 2 == 0
        ]
        mlx_nic, eth_nic = nics[i % 2]
        data[i] = {
            "mlx": mlx_nic,
            "eth": eth_nic,
            "address": get_address(eth_nic),
        }

    return data


def detect_NIC(device_id):
    device_id = get_real_device_id(device_id)
    with open("/tmp/rdma_link.json", "r") as f:
        data = json.load(f)
    return data[str(device_id)]


RNDV_THRESH = 8192


def set_NIC(device_id, init_ucx=True):
    info = detect_NIC(device_id)
    rdma_nic, addr = info["mlx"], info["address"]
    if init_ucx:
        ucp.init(
            options={
                "NET_DEVICES": f"{rdma_nic}:1",
                "TLS": "rc_mlx5,cuda",
                "RNDV_THRESH": str(RNDV_THRESH),
            },
            blocking_progress_mode=True,
        )
        device = torch.device(f"cuda:{device_id}")
        am_allocator = partial(torch.zeros, dtype=torch.uint8, device=device)
        ucp.register_am_allocator(am_allocator, "cuda")
    return addr


async def call_kv_method(engine, method: str, *args, lock=False, **kwargs):
    assert engine and engine.is_running, "Engine is not running"
    driver = engine.engine.model_executor.driver_worker
    coros = [make_async(driver.execute_kv_method)(method, *args, **kwargs)]
    if hasattr(engine.engine.model_executor, "workers"):
        workers = engine.engine.model_executor.workers
        coros += [
            asyncio.wrap_future(
                worker.execute_kv_method.options(
                    concurrency_group="lock"
                ).remote(method, *args, **kwargs)
                if lock
                else worker.execute_kv_method.remote(method, *args, **kwargs)
            ).future()
            for worker in workers
        ]

    return await asyncio.gather(*coros)


def make_done_callback(
    tasks: Set[asyncio.Task], call_back: Callable, *args, **kwargs
):
    def _wrapper(task: asyncio.Task):
        tasks.discard(task)
        call_back(*args, **kwargs)

    return _wrapper


def wrap_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Wrap a tensor with a view when their dtype is not compatible."""
    supported = [
        torch.complex64,
        torch.complex128,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]
    if tensor.dtype not in supported:
        new_dtype = next(
            dtype
            for dtype in supported
            if tensor.dtype.itemsize == dtype.itemsize
        )
        return tensor.view(new_dtype)
    return tensor
