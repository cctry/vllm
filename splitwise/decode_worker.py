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

    async def start_kv_cahce_comm(self, prefilled_seq: SequenceGroup, dst_block_ids: List[int]):
        """ Start KV cache transfer on TP workers
        1. Notify the prefill worker to start sending blocks
        2. Let TP workers start to listen for blocks
        """
        request_id = prefilled_seq.request_id
        url = f"http://{self.prefill_addr}:{self.prefill_port}/kv_cache"
        async with client_session.post(url, json={"request_id": request_id}) as response:
            assert response.ok
        # TODO



# global states for this perfill worker
engine = None
prefill_workers: Dict[str, PrefillWorker] = []
app = FastAPI()
client_session = aiohttp.ClientSession()

async def start_listen_for_blocks(block_ids: List[int]):
    pass

async def pull_kv_cache(prefilled_seq: SequenceGroup, prefill_worker: PrefillWorker):
    request_id = prefilled_seq.request_id
    block_manager = engine.engine.scheduler.block_manager
    # we assume there is only one seq in the sequence group (prompt)
    seq = prefilled_seq.get_seqs()[0]
    dst_block_ids = block_manager.get_block_table(seq)
    # TODO: We should check if the number of blocks are the same
    prefill_worker = prefill_workers[request.client.host]
    await prefill_worker.start_kv_cahce_comm(prefilled_seq, dst_block_ids)


def deserialize_seq_group(data: bytes) -> SequenceGroup:
    data = base64.b64decode(seq_group)
    return pickle.loads(data)

async def generate(stream: AsyncGenerator[RequestOutput], prefill_worker: PrefillWorker):
    async for request_output in stream:
        final_output = request_output
    assert final_output is not None
    await prefill_worker.send_results(final_output)

@app.post("/decode")
async def decode(request: Request) -> Response:
    """Receive prefilled results from prefill workers.
    """
    if not engine.is_running:
        engine.start_background_loop()
    data = await request.json()
    prefilled_seq = await make_async(deserialize_seq_group)(data)
    prefill_worker = prefill_workers[request.client.host]
    request_id = prefilled_seq.request_id
    # allocate blocks
    engine.engine.scheduler._allocate_and_set_running(prefilled_seq)
    await pull_kv_cache(prefilled_seq, prefill_worker)
    # resume the prefill process
    stream = AsyncStream(request_id)
    engine._request_tracker._request_streams[request_id] = stream
    engine.engine.scheduler.waiting.append(prefilled_seq)
    asyncio.get_running_loop().create_task(generate(stream, prefill_worker))
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
    engine.set_decode_worker()

    app.root_path = args.root_path
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
