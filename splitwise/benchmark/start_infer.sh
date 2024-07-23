export RAY_NUM_CPUS=64
# export VLLM_ATTENTION_BACKEND=XFORMERS
python api_server.py --model /data/mistral-7b-instruct-v0_2 -tp 8 --enforce-eager --disable-custom-all-reduce #--enable-chunked-prefill
