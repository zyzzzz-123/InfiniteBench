#!/bin/bash
# pip install vllm -U
# pip install transformers -U
# export PATH=/home/jeeves/.local/bin:$PATH
cd src/
# model_path=$1
# model_name=$2
model_path=/mnt/models/yuzhe/phi3-mini-128k-new/
# model_path=/mnt/models/gaozhang/llama3_8b/
# model_path=/mnt/models/yuzhe/llama3.1-instruct/
# model_name=phi-3-1-longrope-static-m5-3-new-swa128k-bf16-search-short-rope4_5-long-dm-phi3-version
model_name=phi3-mini-128k-new
# code_debug code_run kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find number_string passkey
# for task in code_debug code_run kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find number_string passkey; do
#     python eval_chatglm.py --task ${task} --model_path ${model_path} --output_dir /mnt/yuzhe/Infinite_results --model_name ${model_name} --verbose
# done
# for task in  longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng  math_find number_string passkey; do
#     python eval_chatglm.py --task ${task} --model_path ${model_path} --output_dir /mnt/yuzhe/Infinite_results --model_name ${model_name} --verbose
# done
# for task in longbook_sum_eng; do
#     python eval_chatglm.py --task ${task} --model_path ${model_path} --output_dir /mnt/yuzhe/Infinite_results --model_name ${model_name}
# done

# for task in code_debug; do
#     python compute_scores.py --task ${task} --model_path ${model_path} --output_dir /mnt/yuzhe/Infinite_results --model_name ${model_name}
# done
for task in code_debug code_run kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_calc math_find number_string passkey; do
    python compute_scores.py  --task ${task} --model_path ${model_path} --output_dir /mnt/yuzhe/Infinite_results --model_name ${model_name} --verbose
done