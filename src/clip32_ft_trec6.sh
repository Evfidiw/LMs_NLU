cd "$(dirname "$0")"/..

python clip/main.py \
--data_dir './data/trec6' \
--model_name "./clip-vit-base-patch32" \
--seed 3407 \
--epochs 5 \
--save_model \
--train \
--test
