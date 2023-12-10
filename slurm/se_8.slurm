#!/bin/sh
#SBATCH --job-name=se-8
#SBATCH --output=se-8.out
#SBATCH --error=se-8.out
#SBATCH --nodelist=n53
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all

#set -x
corpus=$1
mt=$2
lr=5e-5
neal=2
alpha=0.5
nr=1

if [ "$corpus" = "drugprot" ]; then
  cn=drugprot
  nl=14
  ne=20
elif [ "$corpus" = "chemprot" ]; then
  cn=chemprot_blurb
  nl=6
  ne=60
elif [ "$corpus" = "bbrel" ]; then
  cn=bbrel_full
  nl=2
  ne=60
else
  printf "Unknown corpus: Specify the number of classes manually. \n"
fi

printf "run on %s (number of classes=%i) \n" "$corpus" "$nl"

for seed in 42 51 61 93 96
do
  echo $nr-th run \; seed : $seed
  srun python3 main.py --dataset_name $cn --model_type $mt --output_dir ./ckpts/ --num_labels $nl --num_epoches $ne --seed $seed --warmup --learning_rate ${lr} --run_id $nr \
                       --num_extra_attention_layers $neal --alpha $alpha --no_test 
  nr=$((nr+1)) 
done
