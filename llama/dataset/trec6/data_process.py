import json
import os
import argparse
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='trec6', type=str)
    parser.add_argument("--input_path", default='./data/trec6/test.txt', type=str)
    parser.add_argument("--output_dir", default='./dataset/trec6', type=str)
    parser.add_argument("--text_json", default='text.json', type=str)
    parser.add_argument("--label_json", default='label.json', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    text_path = os.path.join(args.output_dir, args.text_json)
    label_path = os.path.join(args.output_dir, args.label_json)
    os.makedirs(args.output_dir, exist_ok=True)

    if "trec6" in args.data_dir:
        classes = ['ABBR', 'ENTY', 'DESC', 'HUM', 'LOC', 'NUM']

    with open(args.input_path, 'r', encoding='utf-8') as input_file1, open(text_path, 'w', encoding='utf-8') as output_file1:
        for line in input_file1:
            class_name = line.split(':')[0]

            second_part = line.split(':')[1]
            rest_of_the_string = second_part.split()[1:]
            data_input = ' '.join(rest_of_the_string)

            json_line = json.dumps({
                "instruction": "Identify the label of this sentence. The possible labels are: " + ", ".join(classes) +
                               ". Don't respond anything else, just respond in the {'<identified_class>'} format.",
                "input": data_input,
            })
            output_file1.write(json_line + '\n')

    with open(args.input_path, 'r', encoding='utf-8') as input_file2, open(label_path, 'w', encoding='utf-8') as output_file2:
        for line in input_file2:
            class_name = line.split(':')[0]

            second_part = line.split(':')[1]
            rest_of_the_string = second_part.split()[1:]
            data_input = ' '.join(rest_of_the_string)
            json_line = json.dumps({"output": str(class_name)}, ensure_ascii=False)
            output_file2.write(json_line + '\n')

    print("done!")
