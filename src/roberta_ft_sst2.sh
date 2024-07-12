cd "$(dirname "$0")"/..

python bert/main.py \
--data_dir './data/sst2' \
--model_name "./roberta-base" \
--tokenizer_name "./roberta-base" \
--seed 3407 \
--epochs 5 \
--save_model \
--train \
--test
