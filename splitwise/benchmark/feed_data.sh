export RAY_NUM_CPUS=64
num_req=100
tp=8

model_path=/data
model_name=Meta-Llama-3-70B

declare -a inputs=(
  "1024 50"
  "2048 100"
  "4096 150"
  "8192 200"
  "16384 512"
  "32768 512"
#   "65536 512"
#   "365536 512"
# "65536 512"
# "32768 512"
# "32768 2"
# "1024 1024"
)


for qps in 1.25 1.5 1.75 2
do
    for input in "${inputs[@]}"
    do
        prompt=$(echo $input | cut -d' ' -f1)
        resp=$(echo $input | cut -d' ' -f2)
        out_file="${model_name}_$tp"
        
        message="${tp}_prompt_${prompt}_${resp}_${qps}"
        python test.py \
            --model "$model_path/$model_name" \
            --text-file pg34283.txt \
            --response-length $resp \
            --prompt-length $prompt \
            --num-request $num_req \
            --qps $qps \
            --port 8000 \
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
        sleep 1
    done
done