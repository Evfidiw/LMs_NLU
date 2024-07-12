cd "$(dirname "$0")"/..

python clip/main.py \
--data_dir './data/sst2' \
--model_name "./clip-vit-base-patch32" \
--seed 3407 \
--test
