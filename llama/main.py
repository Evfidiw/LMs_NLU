import pdb
import os
import sys
import json
import argparse
import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser, GenerationConfig, BitsAndBytesConfig

from utils.general_utils import get_model_tokenizer, get_model_name, seed_torch
from utils.prompter import Prompter


@dataclass
class InferArguments:
    model_name: str = field(default='llama', metadata={"help": "model name."})
    model_name_or_path: str = field(default='/play/model/llama-2-13b-chat-hf')
    input_file: str = field(default='test.json', metadata={"help": "input file."})
    output_file: str = field(default='output.json', metadata={"help": "Path to the output file."})
    prompt_template_name: str = field(default='alpaca', metadata={"help": "Prompt template name."})
    bf16: bool = field(default=False, metadata={"help": "The type of data used in the calculation."})
    fp16: bool = field(default=False, metadata={"help": "The type of data used in the calculation."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,  metadata={"help": "Compress the quantization statistics."})
    quant_type: str = field(default="nf4", metadata={"help": "Should be one of `fp4` or `nf4`."})


def inference(options):
    seed_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prompter = Prompter(options.prompt_template_name)
    model_name = get_model_name(options.model_name)
    device_setting = "auto"
    model_class, tokenizer_class, _ = get_model_tokenizer(model_name)
    tokenizer = tokenizer_class.from_pretrained(
        options.model_name_or_path,
        trust_remote_code=True,
    )
    compute_dtype = (torch.float16 if options.fp16 else (torch.bfloat16 if options.bf16 else torch.float32))

    model = model_class.from_pretrained(
        options.model_name_or_path,
        # load_in_4bit=options.bits == 4,
        # load_in_8bit=options.bits == 8,
        device_map=device_setting,
        offload_folder="./cache",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=options.bits == 4,
            load_in_8bit=options.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=options.double_quant,
            bnb_4bit_quant_type=options.quant_type,
        ),
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model.config.torch_dtype = compute_dtype
    if model_name == "llama":
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    elif model_name == "mistral":
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0,
        top_p=0.9,
        max_new_tokens=30,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            # pad_token_id=tokenizer.pad_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                **kwargs,
            )
        generation_output = generation_output.sequences[0]
        output = tokenizer.decode(generation_output, skip_special_tokens=True)
        output = prompter.get_response(output)
        return output

    records = []

    with open(options.input_file, "r") as reader:
        for line in reader:
            data = json.loads(line)
            records.append(data)

    with open(options.output_file, "w") as writer:
        for i, record in enumerate(records):
            result = evaluate(instruction=record['instruction'], input=record['input'])
            print(i+1, result)
            record['output'] = result
            writer.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    hfparser = HfArgumentParser((InferArguments,))
    options, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    inference(options)
