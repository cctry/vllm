export RAY_NUM_CPUS=64
python /data/vllm/vllm/entrypoints/api_server.py --model /data/mistral-7b-instruct-v0_2 -tp 2 --enforce-eager --disable-custom-all-reduce