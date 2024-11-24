cd "$(dirname "$0")"/..

python main.py \
--model_name_or_path "./model" \
--model_name 'llama' \
--input_file "./dataset/sst2/text.json" \
--output_file "./dataset/sst2/output.json" \
--fp16
