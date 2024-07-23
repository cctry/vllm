export RAY_NUM_CPUS=64
resp=100
prompt=2048
num_req=200
qps=1
tp=8

python test.py --model /data/mistral-7b-instruct-v0_2 --text-file prompt.txt --response-length $resp --prompt-length $prompt --num-request $num_req --qps $qps --port 8000

grep '^\[test\]' 7B_tp${tp}_base.txt > base_7B_tp${tp}_prompt${prompt}_resp${resp}_qps${qps}_request.txt
grep '^INFO .* metrics.py:341' 7B_tp${tp}_base.txt > base_7B_tp${tp}_prompt${prompt}_resp${resp}_qps${qps}_token.txt