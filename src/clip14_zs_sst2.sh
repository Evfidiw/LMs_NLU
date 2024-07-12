cd "$(dirname "$0")"/..

python clip/main.py \
--data_dir './data/sst2' \
--model_name "./clip-vit-large-patch14" \
--seed 3407 \
--test
