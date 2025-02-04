#!/bin/bash
## declare posf args arr
declare -a arr=("nill" "piecewise_linear_mask" "piecewise_constant_mask" "piecewise_harmonic_mask" "piecewise_quadratic_mask" "piecewise_sqrt_mask" "piecewise_exponential_mask" "piecewise_sigmoid_mask" "piecewise_tanh_mask" "piecewise_cosine_mask" "piecewise_gaussian_mask")
readonly NEPOCHS=30
RESULT=""
## loop through arr
for i in "${arr[@]}"
do
    echo "Running $i with $NEPOCHS epochs..."
    CONFIG=$(echo "$i")
    OUTPUT=$(CUDA_VISIBLE_DEVICES=0 python3 train_bert.py --model_name senticgcn_bert --dataset rest14 --lr 2e-5 --seed 39 --batch_size 16 --device cuda --num_epoch "$NEPOCHS" --posf "$CONFIG" --mask uniform_aspect_mask_bert --isftext True --nlayers 2 --actf relu)
    EPOCHS=$(echo "$OUTPUT" | grep epoch | tail -1)
    SCORES=$(echo "$OUTPUT" | grep test_acc | tail -1)
    MSG="$CONFIG $EPOCHS $SCORES\n"
    echo -en "$MSG"
    RESULT+="$MSG"
done
DT=$(date '+%Y%m%d-%H%M%S')
FNAME="posf_logs/posf_$DT.txt"
echo -en "$RESULT" > "$FNAME"
echo "Done. Results saved to $FNAME."