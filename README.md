# Objective

Analyze impact of NLP adversarial attacks.

## Attention Distribution

Explore how attention distribution over sentences changes before and after adversarial attacks.


### Note to self

The `textattack` module causes errors when running on cclake hpc cpus. To avoid the error I comment out the line `import pycld2 as cld2` in `venv-cpu/lib/python3.7/site-packages/textattack/shared/utils/strings.py`

## Steps

Example here is for rotten tomatoes dataset

1) 
cd src/data/preprocess
