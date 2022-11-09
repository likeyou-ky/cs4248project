#!/bin/bash
## declare posf args arr
declare -a arr=("uniform_" "normal_" "constant_" "ones_" "zeros_" "eye_" "dirac_" "xavier_uniform_" "xavier_normal_" "kaiming_uniform_" "kaiming_normal_" "trunc_normal_" "orthogonal_" "sparse_")

readonly NEPOCHS=30
RESULT=""
## loop through arr
for i in "${arr[@]}"
do
    echo "Running $i with $NEPOCHS epochs..."
    CONFIG=$(echo "$i")
    OUTPUT=$(CUDA_VISIBLE_DEVICES=0 python3 train_bert.py --model_name senticgcn_bert --initializer "$CONFIG" --dataset rest14 --lr 2e-5 --seed 39 --batch_size 16 --device cuda --num_epoch "$NEPOCHS" --posf "piecewise_gaussian_mask")
    EPOCHS=$(echo "$OUTPUT" | grep epoch | tail -1)
    SCORES=$(echo "$OUTPUT" | grep test_acc | tail -1)
    MSG="$CONFIG $EPOCHS $SCORES\n"
    echo -en "$MSG"
    RESULT+="$MSG"
done
DT=$(date '+%Y%m%d-%H%M%S')
FNAME="improvement_logs/init_$DT.txt"
echo -en "$RESULT" > "$FNAME"
echo "Done. Results saved to $FNAME."