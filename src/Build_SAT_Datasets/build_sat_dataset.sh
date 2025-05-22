#!/bin/bash
cd ./src/Build_SAT_Datasets

PARAMETERS=(
    "3 5 20"
)

N_SAMPLE=520


ID=0
for params in "${PARAMETERS[@]}";do
    N_SAT=$(echo $params | awk '{print $1}')
    K=$(echo $params | awk '{print $2}')
    LENGTH=$(echo $params | awk '{print $3}')
    RESULT_DIR="./Datasets/Seed/Seed_Datasets_${N_SAT}_${K}_${LENGTH}_${N_SAMPLE}/train"
    PROMPT_DIR="./Datasets/Train/Train_prompt_${N_SAT}_${K}_${LENGTH}_${N_SAMPLE}"

    echo "n_sat=$N_SAT, k=$K, length=$LENGTH, n_sample=$N_SAMPLE, result_dir=$RESULT_DIR"

    # build.py
    echo "running single_clause.py ..."
    python ./single_clause.py --n_sat $N_SAT --k $K --result_dir $RESULT_DIR

    # combine.py
    echo "running combine.py ..."
    python ./combine.py --n_sat $N_SAT --k $K --length $LENGTH --n_sample $N_SAMPLE --result_dir $RESULT_DIR

    # prompt.py
    echo "running train_prompt.py ..."
    echo "ID=$ID"
    python ./train_prompt.py --n_sat $N_SAT --k $K --length $LENGTH --n_sample $N_SAMPLE --start_id $ID \
        --seed_datasets_dir $RESULT_DIR --prompt_dir $PROMPT_DIR

    echo "n_sat=$N_SAT, k=$K, length=$LENGTH, n_sample=$N_SAMPLE, result_dir=$RESULT_DIR done!"
    ID=$((ID + N_SAMPLE))

done

echo "all done!"