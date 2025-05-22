# Step 1: Define variables
model_path= # TODO
model_name= # TODO

lora_name=$model_name
lora_path=""

num_sample=100 # Number of samples per granularity in the test set
n_pass=12 # n for pass@k
temperature=0.6

PARAMETERS=(
    "5 8 40"
    "6 7 40"
    "4 7 20"
    "4 8 40"
    "4 7 40"
    "3 7 20"
    "3 6 20"
    "3 5 20"
    "3 5 25"
    "3 7 40"
)

tensor_parallel_size=4
port=8050

docker_name="vllm_model_${model_name}"

# docker kill $docker_name || true
# docker rm $docker_name || true

# Step 2: Start Docker container
docker run -d --gpus '"device=4,5,6,7"' --name ${docker_name} \
  -v /nfs100:/nfs100 \
  -v /models:/models \
  -p ${port}:8000 \
  --shm-size=8g \
  vllm:250103 \
  --model=${model_path} \
  --served-model-name=${model_name} \
  --tensor-parallel-size=${tensor_parallel_size} \
  --max-num-seqs=64 \
  --gpu-memory-utilization=0.90 \
  --max-model-len=16384
  # --enable-lora \
  # --lora-modules "${lora_name}=${lora_path}" \
  # --max_lora_rank=32
# 
sleep 180
echo "Docker ${docker_name} started successfully"


# Step 3: Obtain the IP address of the container and construct vllm_url
container_ip=$(docker inspect -f '{{.NetworkSettings.Networks.bridge.IPAddress}}' ${docker_name})
if [[ -z "${container_ip}" ]]; then
  echo "Failed to obtain container IP address, script terminated."
  exit 1
fi
vllm_url="http://${container_ip}:8000/v1/completions"
echo "vllm_url: ${vllm_url}"


# Step 4: Loop through each parameter combination
for params in "${PARAMETERS[@]}"; do
    n_sat=$(echo $params | awk '{print $1}')
    k=$(echo $params | awk '{print $2}')
    length=$(echo $params | awk '{print $3}')

    echo "work: n_sat=$n_sat, k=$k, length=$length"

    prompt_file="./data/test/n${n_sat}_k${k}_length${length}_sample${num_sample}.jsonl"
    work_dir="./results/n${n_sat}_k${k}_length${length}_sample${num_sample}"
    docker_name="SAT_${model_name}"
    docker run --rm -it --name ${docker_name} \
        -v /nfs100:/nfs100 \
        -v /dev/shm:/dev/shm \
        -v /models:/models \
        trl_env:0910 sh -c "python ./src/sample/inference_k.py \
        --work_dir=${work_dir} \
        --model_name=${lora_name} \
        --model_path=${model_path} \
        --prompt_file=${prompt_file} \
        --num_samples=${n_pass} \
        --vllm_url=${vllm_url} \
        --temperature=${temperature} && \
        python ./src/sample/eval_k.py \
        --work_dir=${work_dir} \
        --model_path=${model_path} \
        --model_name=${lora_name}"
done

echo "ALL Done!"