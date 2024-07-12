import os
import sys
import json
import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser, GenerationConfig, BitsAndBytesConfig
import pdb
from openai import OpenAI


@dataclass
class InferArguments:
    model_name: str = field(default='llama', metadata={"help": "model name."})
    input_file: str = field(default='test.json', metadata={"help": "input file."})
    output_file: str = field(default='output.json', metadata={"help": "Path to the output file."})


def inference(options):
    api_key = "sk-m3frVjNHs0rSvUNt0aAa5e7220F24a42A6844a9f9336BdDe"
    base_url = "https://api.zyai.online/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = options.model_name
    print(model)

    def evaluate(
        prompts,
    ):
        messages = [
            {"role": "user", "content": prompts},
        ]
        completion = client.chat.completions.create(
            model=model,  # "gpt-3.5-turbo", "gpt-4", "gpt-4o"
            messages=messages,
            temperature=0,
            max_tokens=200,
            top_p=0.99,
        )
        res_msg = completion.choices[0].message.content.strip()
        return res_msg

    records = []

    with open(options.input_file, "r") as reader:
        for line in reader:
            data = json.loads(line)
            records.append(data)

    with open(options.output_file, "w") as writer:
        for i, record in enumerate(records):
            result = evaluate(record["input"])
            print(i+1, result)
            output_record = {'output': result}
            writer.write(json.dumps(output_record, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    hfparser = HfArgumentParser((InferArguments,))
    options, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    inference(options)
