rm -f output
for s in 38 26 42 50 1001 12 122 4248 7 49
do
  echo "processing seed $s"
  echo "processing seed $s" >> output
  CUDA_VISIBLE_DEVICES=0 python train_bert.py --model_name senticgcn_bert --dataset rest14 --lr 2e-5 --seed "$s" --batch_size 16 >> output
done
