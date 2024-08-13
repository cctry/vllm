export RAY_NUM_CPUS=64
num_req=50
tp=8

model_path=/data
model_name=Meta-Llama-3-70B-dummy

port=8000

declare -a inputs=(
#   "1024 50"
#   "2048 100"
#   "4096 150"
#   "8192 200"
#   "16384 512"
  "32768 512"
#   "65536 512"
)


for qps in 1 # 0.25 0.5 0.75 1 1.5 2
do
    for input in "${inputs[@]}"
    do
        prompt=$(echo $input | cut -d' ' -f1)
        resp=$(echo $input | cut -d' ' -f2)
        out_file="${model_name}_${tp}_pull"
        
        message="${tp}_prompt_${prompt}_${resp}_${qps}"
        python test.py \
            --model "$model_path/$model_name" \
            --text-file pg57126.txt \
            --response-length $resp \
            --prompt-length $prompt \
            --num-request $num_req \
            --qps $qps \
            --port $port \
            --message $message

       
        grep "^\[test${message}\]" ${out_file}.txt > ${out_file}_prompt${prompt}_resp${resp}_qps${qps}_request.txt
        grep '^INFO .* metrics.py:341' ${out_file}.txt > ${out_file}_prompt${prompt}_resp${resp}_qps${qps}_token.txt

        start_line=$(grep -n "^\[test${message}\]" ${out_file}.txt | head -1 | cut -d: -f1)
        sed -n "${start_line},\$p" ${out_file}.txt > temp_extracted.txt
        grep '^INFO .* async_llm_engine.py:247' temp_extracted.txt > ${out_file}_prompt${prompt}_resp${resp}_qps${qps}_iteration.txt
        rm temp_extracted.txt

        if [ $? -ne 0 ]; then
            echo "Error at prompt $prompt resp $resp qps $qps."
            break
        fi
        echo "$prompt resp $resp qps $qps finished"
        sleep 3
    done
done