#!/bin/sh
#SBATCH --job-name=mts-10
#SBATCH --output=mts_10.out
#SBATCH --error=mts_10.out
#SBATCH --nodelist=n101
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all

#set -x
corpus=drugprot
mt=mts
lr=1e-4
neal=1
alpha=1.0
nr=4

if [ "$corpus" = "drugprot" ]; then
  cn=drugprot
  nl=14
  ne=10
elif [ "$corpus" = "chemprot" ]; then
  cn=chemprot_blurb
  nl=6
  ne=30
elif [ "$corpus" = "bbrel" ]; then
  cn=bbrel_full
  nl=3
  ne=60
else
  printf "Unknown corpus: Specify the number of classes manually. \n"
fi

printf "run on %s (number of classes=%i) \n" "$corpus" "$nl"

for seed in 93
do
  echo $nr-th run \; seed : $seed
  srun python3 main.py --dataset_name $cn --model_type $mt --output_dir ./ckpts/ --num_labels $nl --num_epoches $ne --seed $seed --warmup --shuffle_train --learning_rate ${lr} --run_id $nr \
                       --num_extra_attention_layers $neal --alpha $alpha --no_test --batch_size 8 
  nr=$((nr+1)) 
done
