export CUDA_VISIBLE_DEVICES=2

MODEL= # TODO
OUTPUT_DIR=./results/$MODEL

NUM_GPUS=1
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# AMC 22/23
python ./scripts/test/math_eval/test_amc.py \
    --model_path $MODEL \
    --output_dir $OUTPUT_DIR

lighteval vllm $MODEL_ARGS "lighteval|aime24|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details 

lighteval vllm $MODEL_ARGS "lighteval|aime25|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details 

lighteval vllm $MODEL_ARGS "lighteval|math_500|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details 

lighteval vllm $MODEL_ARGS "lighteval|gpqa:diamond|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details 

lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# If Needed
#python ./scripts/test/frequency_cloud.py \
#    --work_dir $OUTPUT_DIR \
#    --model $MODEL
