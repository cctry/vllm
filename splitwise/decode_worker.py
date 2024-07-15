import argparse
import asyncio
import os
from typing import Dict

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from test_stub import get_prefill_worker
from utils import call_kv_method, deserialize_seq_group

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sequence import SequenceGroup, SequenceStatus
from vllm.usage.usage_lib import UsageContext
from vllm.utils import make_async, random_uuid

# global states for this perfill worker
engine = None
app = FastAPI()
http_session: aiohttp.ClientSession


class PrefillWorker:
    def __init__(self, prefill_addr: str, prefill_port: int):
        self.prefill_addr = prefill_addr
        self.prefill_port = prefill_port

    async def send_prefill_request(
        self, http_session: aiohttp.ClientSession, request, request_id
    ):
        prefill_request = {
            "request_id": request_id,
            **request,
        }
        url = f"http://{self.prefill_addr}:{self.prefill_port}/prefill"
        async with http_session.post(url, json=prefill_request) as response:
            response.raise_for_status()
            data = await response.json()
            return data


async def pull_kv_cache(prefilled_seq: SequenceGroup, kv_server_info: Dict):
    request_id = prefilled_seq.request_id
    block_manager = engine.engine.scheduler.block_manager
    # we assume there is only one seq in the sequence group (prompt)
    seq = prefilled_seq.get_seqs()[0]
    dst_block_ids = block_manager.get_block_table(seq)
    # TODO: We should check if the number of blocks are the same
    await call_kv_method(
        engine, "pull_kv", request_id, dst_block_ids, kv_server_info
    )


async def decode(prefilled_seq: SequenceGroup, kv_server_info: Dict):
    """Receive prefilled results from prefill workers."""
    request_id = prefilled_seq.request_id
    # allocate blocks
    for seq in prefilled_seq.get_seqs():
        # vllm wants them to be waiting
        seq.status = SequenceStatus.WAITING
    engine.engine.scheduler.block_manager.mark_blocks_as_computed(prefilled_seq)
    engine.engine.scheduler._allocate_and_set_running(prefilled_seq)
    await pull_kv_cache(prefilled_seq, kv_server_info)
    # resume the inference process
    stream = AsyncStream(request_id)
    engine._request_tracker._request_streams[request_id] = stream
    engine.engine.scheduler.running.append(prefilled_seq)
    engine._request_tracker.new_requests_event.set()
    final_output = None
    async for request_output in stream:
        final_output = request_output
    assert final_output is not None
    prompt = final_output.prompt
    return [prompt + output.text for output in final_output.outputs]


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    request_id = random_uuid()
    assert "request_id" not in request_dict, "The request contains request ID"
    addr, host = get_prefill_worker()
    comm = PrefillWorker(addr, host)
    data = await comm.send_prefill_request(
        http_session, request_dict, request_id
    )
    prefilled_seq = await make_async(deserialize_seq_group)(data["seq_group"])
    kv_server_info = data["kv_server_info"]
    text = await decode(prefilled_seq, kv_server_info)
    return JSONResponse({"text": text})


async def run(config: uvicorn.Config):
    global http_session, engine
    engine.start_background_loop()
    await call_kv_method(engine, "decode_kv_init")
    http_session = aiohttp.ClientSession()
    server = uvicorn.Server(config=config)
    await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log-level", type=str, default="debug")
    parser.add_argument("--model-path", type=str, default="/data/mistral-7b-instruct-v0_2")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    # engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args = AsyncEngineArgs(
        model=args.model_path,
        tensor_parallel_size=2,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        engine_use_ray=False, # Must be False so we can access the scheduler
    )
    os.environ["RAY_NUM_CPUS"] = "64"
    os.environ["WORKER_MODULE"] = "worker"
    os.environ["WORKER_CLASS"] = "WorkerSplitwise"

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER
    )

    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    asyncio.run(run(config))
