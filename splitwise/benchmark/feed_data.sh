export RAY_NUM_CPUS=64
resp=512
prompt=32768
num_req=100
qps=2
tp=8
message="${tp}_prompt_${prompt}_${resp}_${qps}"

python test.py --model /maas-us/models/Mixtral-8x7B-v0.1 --text-file prompt.txt --response-length $resp --prompt-length $prompt --num-request $num_req --qps $qps --port 8000

out_file="base_8x7B_tp${tp}"

grep "^\[test${message}\]" ${out_file}.txt > ${out_file}_prompt${prompt}_resp${resp}_qps${qps}_request.txt
grep '^INFO .* metrics.py:341' ${out_file}.txt > ${out_file}_prompt${prompt}_resp${resp}_qps${qps}_token.txt

# start_line=$(grep -n "^\[test${message}\]" ${out_file}.txt | head -1 | cut -d: -f1)
# sed -n "${start_line},\$p" ${out_file}.txt > temp_extracted.txt
# grep '^INFO .* metrics.py:341' temp_extracted.txt > ${out_file}_prompt${prompt}_resp${resp}_qps${qps}_token.txt
# rm temp_extracted.txt