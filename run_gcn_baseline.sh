# CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name senticgcn --dataset rest14 --save True --learning_rate 1e-3 --seed 39 --batch_size 16 --hidden_dim 300 --num_epoch 30 --posf piecewise_linear_mask
rm -f output
for s in 38 26 42 50 1001 12 122 4248 7 49 76 52 84 100 2002 24 244 8496 14 98
do
  echo "processing seed $s"
  echo "processing seed $s" >> output
  CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name baselinegcn --dataset rest14 --save True --learning_rate 1e-3 --seed "$s" --batch_size 16 --hidden_dim 300
done