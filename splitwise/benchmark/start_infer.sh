export RAY_NUM_CPUS=64
# export VLLM_ATTENTION_BACKEND=XFORMERS
model_path=/data
model_name=Meta-Llama-3.1-70B-Instruct
TP=8
# python api_server.py --model $model_path/$model_name  -tp $TP --enforce-eager --disable-custom-all-reduce #--enable-chunked-prefill 

python api_server.py --model "$model_path/$model_name" \
    --enforce-eager --disable-custom-all-reduce \
    -tp "$TP" > "${model_name}_$TP.txt"
