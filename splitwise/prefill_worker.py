import argparse
import asyncio
import os
import time
from typing import List, Set

import aiohttp
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from utils import call_kv_method, make_done_callback, serialize_seq_group

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroup
from vllm.usage.usage_lib import UsageContext
from vllm.utils import make_async

# global states for this perfill worker
engine = None
app = FastAPI()
client_session: aiohttp.ClientSession
background_tasks: Set[asyncio.Task] = set()


def free_blocks(blocks):
    block_manager = engine.engine.scheduler.block_manager
    block_manager._free_block_table(blocks)


def push_kv_cache(request_id: str, blocks):
    block_ids = [block.block_number for block in blocks]
    task = asyncio.create_task(
        call_kv_method(engine, "push_kv", request_id, block_ids)
    )
    background_tasks.add(task)
    task.add_done_callback(
        make_done_callback(background_tasks, free_blocks, blocks)
    )


@app.post("/prefill")
async def prefill(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - request_id: the request id.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    request_id = request_dict.pop("request_id")
    sampling_params = SamplingParams(**request_dict)

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        if engine.is_prefill_worker:
            assert final_output is None, "Only one output is expected"
        final_output = request_output
        print(
            "Complete at",
            time.time(),
            [output.text for output in final_output.outputs],
        )

    assert final_output is not None
    seq_group = final_output.seq_group
    blocks = final_output.blocks
    # This will start the push in the background
    push_kv_cache(request_id, blocks)
    seq_group_data = await make_async(serialize_seq_group)(seq_group)
    return JSONResponse(
        {"seq_group": seq_group_data, "kv_server_info": kv_info}
    )


async def run(config: uvicorn.Config):
    global client_session, engine, kv_info
    engine.start_background_loop()
    engine.set_prefill_worker()
    kv_info = await call_kv_method(engine, "prefill_kv_init", 13337)
    client_session = aiohttp.ClientSession()
    server = uvicorn.Server(config=config)
    await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    # engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args = AsyncEngineArgs(
        model="gpt2",
        tensor_parallel_size=2,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        engine_use_ray=False,
        worker_use_ray=True,
    )

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

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(run(config))
