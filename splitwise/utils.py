import asyncio
import base64
import contextlib
import os
import pickle
import re
import subprocess
import time
import zlib
from contextlib import contextmanager
from functools import partial
from typing import Callable, Set

import torch
import ucp

from vllm.sequence import SequenceGroup
from vllm.utils import make_async


class Recorder:
    def __init__(self, desc):
        self.desc = desc
        self.tagged_time = {}

    @contextmanager
    def record(self, tag):
        start = time.time()
        yield
        end = time.time()
        if tag not in self.tagged_time:
            self.tagged_time[tag] = end - start
        else:
            self.tagged_time[tag] += end - start

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


def detect_NIC(device_id):
    device_id = get_real_device_id(device_id)
    topo_output = subprocess.check_output(
        ["nvidia-smi", "topo", "-m"], text=True
    )
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    topo_output = ansi_escape.sub("", topo_output)
    topo_lines = topo_output.split("\n")
    devices = topo_lines[0].split()[:-4]
    device = devices[device_id]
    conns = topo_lines[device_id + 1].split()[:-2]
    assert conns[0] == device, "Pasing error"
    assert len(conns[1:]) == len(devices), "Parsing error"
    mlxs = [devices[i] for i, conn in enumerate(conns[1:]) if conn == "PXB"]

    rdma_output = subprocess.check_output(["rdma", "link"], text=True)
    interfaces = {}
    pattern = re.compile(
        rf'(?:{"|".join(mlxs)})/\d+.*LINK_UP.*netdev\s+(\w+\d+)'
    )
    number_pattern = re.compile(r"\d+")

    for match in re.finditer(pattern, rdma_output):
        interface_name = match.group(1)
        interface_number = int(number_pattern.search(interface_name).group())
        if interface_number % 2 == 0:  # only even eth interfaces
            target = next(t for t in mlxs if t in match.group())
            interfaces[target] = interface_name

    assert len(interfaces) > 0, "No RDMA interfaces found"
    return list(interfaces.items())[device_id % 2]

RNDV_THRESH = 8192

def set_NIC(device_id, init_ucx=True):
    rdma_nic, eth_nic = detect_NIC(device_id)
    if init_ucx:
        ucp.init(
            options={
                "NET_DEVICES": f"{rdma_nic}:1",
                "TLS": "rc_mlx5,cuda",
                "RNDV_THRESH": str(RNDV_THRESH)
            },
            blocking_progress_mode=True,
        )
        device = torch.device(f"cuda:{device_id}")
        am_allocator = partial(torch.zeros, dtype=torch.uint8, device=device)
        ucp.register_am_allocator(am_allocator, "cuda")
    host = ucp.get_address(ifname=eth_nic)
    return host


async def call_kv_method(engine, method: str, *args, **kwargs):
    assert engine and engine.is_running, "Engine is not running"
    driver = engine.engine.model_executor.driver_worker
    coros = [make_async(driver.execute_kv_method)(method, *args, **kwargs)]
    if hasattr(engine.engine.model_executor, "workers"):
        workers = engine.engine.model_executor.workers
        coros += [
            asyncio.wrap_future(
                worker.execute_kv_method.remote(
                    method, *args, **kwargs
                ).future()
            )
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
