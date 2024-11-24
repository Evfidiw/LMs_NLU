cd "$(dirname "$0")"/..

python main.py \
--model_name_or_path "./model" \
--model_name 'llama' \
--input_file "./dataset/trec6/text.json" \
--output_file "./dataset/trec6/output.json" \
--fp16
