## Introduction

In the era of Large Language Model (LLM), exploring the applications of various language models in Natural Language Understanding (NLU) remains crucial. NLU is a significant task within Natural Language Processing (NLP), with text classification being one of its most representative tasks. 

In this study, we will investigate the capabilities of different language models using some datasets. This approach will enable us to evaluate and compare the performance of these models in handling NLU tasks.

This repository is also the template on the NLU task.



## Preparation 

Install the libraries needed to work:

``````
`pip install -r requirements.txt
``````



## Experiment

The following are the details on the various language models.

- [BERT-based](https://github.com/Evfidiw/LMs_for_NLU/tree/main/bert)
- [CLIP](https://github.com/Evfidiw/LMs_for_NLU/tree/main/clip)
- [GPT](https://github.com/Evfidiw/LMs_for_NLU/tree/main/gpt)



## Refernece Results

#### Zero-shot Results

Metric: accuracy (%)

| Model                         | TREC6 | SST2  |
| ----------------------------- | ----- | ----- |
| CLIP (clip-vit-base-patch32)  | 39.60 | 56.51 |
| CLIP (clip-vit-large-patch14) | 47.20 | 52.88 |
| GPT3.5 (gpt-3.5-turbo)        | 73.00 |       |
| GPT4 (gpt-4)                  | 84.20 |       |



#### Fine-tuning Results

Metric: accuracy (%)

| Model                        | Method        | TREC6 | SST2  |
| ---------------------------- | ------------- | ----- | ----- |
| BERT (bert-base-uncased)     | Full finetune | 96.72 | 96.55 |
| RoBERTa (roberta-base)       | Full finetune | 97.54 | 97.36 |
| CLIP (clip-vit-base-patch32) | Full finetune | 94.60 | 88.30 |



## Analysis

- LLMs demonstrate superior performance in zero-shot scenarios, where they can effectively handle tasks without the need for task-specific training. This makes LLMs particularly advantageous for applications requiring flexibility and adaptability to a wide range of tasks with minimal preparation.
- PLMs, especially those that are encoder-only, such as BERT, are sufficient through finetuning for many NLU tasks. These models excel in understanding and processing natural language, making them suitable for tasks like text classification, sentiment analysis, and named entity recognition.
- The effectiveness of different prompts varies significantly, impacting the performance of language models. The design of prompts can influence performance significantly.



## References

- [pipeline](https://github.com/Evfidiw/LMs_for_NLU/blob/main/experimence.md)