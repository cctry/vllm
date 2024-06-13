import argparse
import json
import ssl
from typing import AsyncGenerator, List, Dict
import asyncio

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.sequence import SequenceGroup
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid, make_async
import pickle
import base64
import aiohttp


def serialize_seq_group(seq_group: SequenceGroup) -> bytes:
    data = pickle.dumps(seq_group)
    return base64.b64encode(data)


class DecodeWorker:
    def __init__(
        self, decode_addr: str, decode_port: int, kv_addr: str, kv_port: int
    ):
        self.decode_addr = decode_addr
        self.decode_port = decode_port
        self.kv_addr = kv_addr
        self.kv_port = kv_port
        # TODO: create ucx connection

    async def send_prefilled_seq(self, prefilled_seq: SequenceGroup):
        data = await make_async(serialize_seq_group)(
            prefilled_seq
        )  # FIXME: This may not correct
        url = f"http://{self.decode_addr}:{self.decode_port}/decode"
        async with client_session.post(
            url, json={"seq_group": data}
        ) as response:
            assert response.ok

    async def start_kv_cahce_comm(self, block_ids: List[int]):
        """Start KV cache transfer on TP workers
        KV cache are stored on TP workers.
        TP workers on decode workers will run a ucx server listening for blocks
        TP workers on prefill workers will send blocks to decode workers
        TODO: Handling different TP
        """
        pass  # TODO


# global states for this perfill worker
engine = None
decode_workers: Dict[str, DecodeWorker] = []
prefilled_seqs: Dict[str, SequenceGroup] = {}
app = FastAPI()
client_session = aiohttp.ClientSession()


@app.post("/kv_cache")
async def kv_cache(request: Request) -> Response:
    """Handle KV cache pulling.

    The request should be a JSON object with the following fields:
    - request_id: request id for the prefilled request
    """
    request_dict = await request.json()
    request_id = request_dict["request_id"]
    block_manager = engine.engine.scheduler.block_manager
    # we assume there is only one seq in the sequence group (prompt)
    seq = prefilled_seqs.pop(request_id).get_seqs()[0]
    block_ids = block_manager.get_block_table(seq)
    decode_worker = decode_workers[request.client.host]
    task = asyncio.get_running_loop().create_task(
        decode_worker.start_kv_cahce_comm(block_ids)
    )
    # free the cache blocks
    # Hopefully, this is compatiable with prefix caching.
    task.add_done_callback(lambda _: block_manager.free(seq))
    return Response(status_code=200)


async def send_prefilled_seq(prefilled_seq: SequenceGroup):
    decode_worker = await get_decode_worker()  # TODO: Mock this function
    await decode_worker.send_prefill_results(prefilled_seq)


@app.post("/prefill")
async def prefill(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        assert final_output is None, "Only one output is expected"
        final_output = request_output

    assert final_output is not None
    seq_group = final_output.seq_group
    await send_prefilled_seq(seq_group)
    prefilled_seqs[request_id] = seq_group

    # TODO: We may need to get the full text from decode worker and then return it to the client
    return Response(status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    # engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args = AsyncEngineArgs(
        model="01-ai/Yi-6B",
        tensor_parallel_size=2,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        engine_use_ray=False,
    )
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER
    )
    engine.set_prefill_worker()

    app.root_path = args.root_path
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
