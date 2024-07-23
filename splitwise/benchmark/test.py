import aiohttp
import asyncio
from transformers import AutoTokenizer
import argparse
import time


async def benchmark(url, payload, qps, num_request):
    interval = 1.0 / qps

    async def send_request(session, i, url, payload):
        try:
            start_time = time.time()
            async with session.post(url, json=payload) as response:
                elapsed = time.time() - start_time
                print(i, response.status, elapsed)
                return (i, response.status, elapsed)
        except Exception as e:
            print(e)
            return (i, None, None, str(e))

    async def producer():
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_request):
                tasks.append(asyncio.create_task(send_request(session, i, url, payload)))
                await asyncio.sleep(interval)
            results = await asyncio.gather(*tasks)
            return results

    results = await producer()
    # for result in results:
    #     print(result)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--text-file", type=str, required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--response-length", type=int, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--qps", type=float, default=1)
    parser.add_argument("--num-request", type=int, default=10)
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}/generate"

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(args.text_file, "r") as f:
        text = f.read()
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]

    if len(input_ids) >= args.prompt_length:
        prompt_id = input_ids[: args.prompt_length]
    else:
        repeat_count = (args.prompt_length + len(input_ids) - 1) // len(input_ids)
        prompt_id = (input_ids * repeat_count)[:args.prompt_length]
    
    prompt = tokenizer.decode(prompt_id)
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": args.response_length,
    }

    asyncio.run(benchmark(url, payload, args.qps, args.num_request))
