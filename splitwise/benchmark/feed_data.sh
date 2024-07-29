export RAY_NUM_CPUS=64
num_req=100
tp=8

model_path=/data
model_name=mistral-7b-instruct-v0_2

declare -a inputs=(
  "1024 50"
  "2048 100"
#   "4096 200"
#   "8192 300" 
#   "16384 512"
#   "365536 512"
)


for qps in 1 2
do
    for input in "${inputs[@]}"
    do
        prompt=$(echo $input | cut -d' ' -f1)
        resp=$(echo $input | cut -d' ' -f2)
        message="${tp}_prompt_${prompt}_${resp}_${qps}"
        python test.py --model "$model_path/$model_name" --text-file prompt.txt --response-length $resp --prompt-length $prompt --num-request $num_req --qps $qps --port 8000 --message $message

        out_file="${model_name}_$tp"

        grep "^\[test${message}\]" ${out_file}.txt > ${out_file}_prompt${prompt}_resp${resp}_qps${qps}_request.txt
        grep '^INFO .* metrics.py:341' ${out_file}.txt > ${out_file}_prompt${prompt}_resp${resp}_qps${qps}_token.txt
    done
done
# start_line=$(grep -n "^\[test${message}\]" ${out_file}.txt | head -1 | cut -d: -f1)
# sed -n "${start_line},\$p" ${out_file}.txt > temp_extracted.txt
# grep '^INFO .* metrics.py:341' temp_extracted.txt > ${out_file}_prompt${prompt}_resp${resp}_qps${qps}_token.txt
# rm temp_extracted.txt