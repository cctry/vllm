import argparse
import json
import ssl
from typing import AsyncIterator, List, Dict
import asyncio

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup, SequenceStatus
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid, make_async
import pickle
import base64
import aiohttp


class PrefillWorker:
    def __init__(
        self, prefill_addr: str, prefill_port: int, kv_addr: str, kv_port: int
    ):
        self.prefill_addr = prefill_addr
        self.prefill_port = prefill_port
        self.kv_addr = kv_addr
        self.kv_port = kv_port
        # TODO: create ucx connection

    async def send_results(self, final_output: RequestOutput):
        prompt = final_output.prompt
        text_outputs = [prompt + output.text for output in final_output.outputs]
        ret = {"text": text_outputs}
        # TODO: should sent this to somewhere

    async def start_kv_cahce_comm(
        self, prefilled_seq: SequenceGroup, dst_block_ids: List[int]
    ):
        """Start KV cache transfer on TP workers
        1. Notify the prefill worker to start sending blocks
        2. Let TP workers start to listen for blocks
        """
        request_id = prefilled_seq.request_id
        url = f"http://{self.prefill_addr}:{self.prefill_port}/kv_cache"
        async with client_session.post(
            url, json={"request_id": request_id}
        ) as response:
            assert response.ok
        # TODO


# global states for this perfill worker
engine = None
prefill_workers: Dict[str, PrefillWorker] = {}
app = FastAPI()
client_session: aiohttp.ClientSession


prefill_workers["127.0.0.1"] = PrefillWorker(
    "127.0.0.1", 8001, "127.0.0.1", 8101
)


async def start_listen_for_blocks(block_ids: List[int]):
    pass


async def pull_kv_cache(
    prefilled_seq: SequenceGroup, prefill_worker: PrefillWorker
):
    block_manager = engine.engine.scheduler.block_manager
    # we assume there is only one seq in the sequence group (prompt)
    seq = prefilled_seq.get_seqs()[0]
    dst_block_ids = block_manager.get_block_table(seq)
    # TODO: We should check if the number of blocks are the same
    await prefill_worker.start_kv_cahce_comm(prefilled_seq, dst_block_ids)


def deserialize_seq_group(data: str) -> SequenceGroup:
    data = base64.b64decode(data)
    return pickle.loads(data)


async def generate(
    stream: AsyncIterator[RequestOutput], prefill_worker: PrefillWorker
):
    async for request_output in stream:
        final_output = request_output
    assert final_output is not None
    await prefill_worker.send_results(final_output)


@app.post("/decode")
async def decode(request: Request) -> Response:
    """Receive prefilled results from prefill workers."""
    data = await request.json()
    prefilled_seq = await make_async(deserialize_seq_group)(data["seq_group"])
    prefill_worker = prefill_workers[request.client.host]
    request_id = prefilled_seq.request_id
    # allocate blocks
    for seq in prefilled_seq.get_seqs():
        # vllm wants them to be waiting
        seq.status = SequenceStatus.WAITING
    engine.engine.scheduler.block_manager.mark_blocks_as_computed(prefilled_seq)
    engine.engine.scheduler._allocate_and_set_running(prefilled_seq)
    await pull_kv_cache(prefilled_seq, prefill_worker)
    # resume the prefill process
    stream = AsyncStream(request_id)
    engine._request_tracker._request_streams[request_id] = stream
    engine.engine.scheduler.waiting.append(prefilled_seq)
    asyncio.get_running_loop().create_task(generate(stream, prefill_worker))
    return Response(status_code=200)


async def run(config: uvicorn.Config):
    global client_session
    engine.start_background_loop()
    client_session = aiohttp.ClientSession()
    server = uvicorn.Server(config=config)
    await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8001)
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
    engine.set_decode_worker()

    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )

    asyncio.run(run(config))
