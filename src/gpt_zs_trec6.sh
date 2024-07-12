cd "$(dirname "$0")"/..

python gpt/main.py \
--input_file "./gpt/dataset/trec6/text.json" \
--output_file "./gpt/dataset/trec6/output.json" \
--model_name "gpt-4"
