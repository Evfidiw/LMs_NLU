cd "$(dirname "$0")"/..

python bert/main.py \
--data_dir './data/trec6' \
--model_name "./bert-base-uncased" \
--tokenizer_name "./bert-base-uncased" \
--seed 3407 \
--epochs 5 \
--save_model \
--train \
--test
