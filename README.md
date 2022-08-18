# Objective

Analyze impact of NLP adversarial attacks.

## Attention Distribution

Explore how attention distribution over sentences changes before and after adversarial attacks.


### Note to self

The `textattack` module causes errors when running on cclake hpc cpus. To avoid the error I comment out the line `import pycld2 as cld2` in `venv-cpu/lib/python3.7/site-packages/textattack/shared/utils/strings.py`

## Steps

Example here is for rotten tomatoes dataset

1) Load data: `cd src/data/preprocess` and then `python prepare_data_files.py --data_name rt --out_dir ../data_files/rt`
2) Train: `python train.py --out_dir experiments/trained_models/rt --model_name bert-base-uncased --data_dir_path src/data/data_files/rt --bs 8 --epochs 4 --lr 1e-5 --seed 1`
3) Attack in batches: `cd experiments` and then `sbatch ./submit_array.sh --model_path ./trained_models/rt/bert-base-uncased_rt_seed1.th --model_name bert-base-uncased --data_dir_path ../src/data/data_files/rt --out_dir_path ../src/data/data_files/rt/attacks/bert/pwws/batch_output --attack_recipe pwws --part test`
4) Combine batch output into single file: `cd src/attack` and then `python batch_to_single_file.py --batch_dir_path ../data/data_files/rt/attacks/bert/pwws/batch_output --out_dir_path ../data/data_files/rt/attacks/bert/pwws --part test --num_files 22 --batch_size 50`
5) Evaluate the attack: `python eval_attack.py --model_path experiments/trained_models/rt/bert-base-uncased_rt_seed1.th --model_name bert-base-uncased --attack_dir_path src/data/data_files/rt/attacks/bert/pwws --part test`
