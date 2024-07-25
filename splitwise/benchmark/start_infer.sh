export RAY_NUM_CPUS=64
# export VLLM_ATTENTION_BACKEND=XFORMERS
python api_server.py --model /maas-us/models/Mixtral-8x7B-v0.1 -tp 8 --enforce-eager --disable-custom-all-reduce #--enable-chunked-prefill
