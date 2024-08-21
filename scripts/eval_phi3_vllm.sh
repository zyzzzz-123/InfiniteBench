#!/bin/bash
# pip install vllm -U
# pip install transformers -U
# export PATH=/home/jeeves/.local/bin:$PATH
cd src/
# model_path=$1
# model_name=$2
model_path=/mnt/models/yuzhe/phi3-mini-128k-new/
# model_path=/mnt/models/gaozhang/llama3_8b/
# model_path=/mnt/models/yuzhe/llama3.1-instruct
model_name=chatglm3
# code_debug code_run kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find number_string passkey
# for task in code_debug code_run kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find number_string passkey; do
#     python eval_chatglm.py --task ${task} --model_path ${model_path} --output_dir /mnt/yuzhe/Infinite_results --model_name ${model_name} --verbose
# done
for task in longbook_sum_eng; do
    python eval_phi3_vllm.py --task ${task} --model_path ${model_path} --output_dir /mnt/yuzhe/Infinite_results --model_name ${model_name}
done