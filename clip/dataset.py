import os
import pdb
import torch
from datasets import Dataset


def data_processing(mode, args):
    data_dir = args.data_dir
    if mode == 'train':
        file_path = os.path.join(data_dir, 'train.txt')
    elif mode == 'dev':
        file_path = os.path.join(data_dir, 'dev.txt')
    elif mode == 'test':
        file_path = os.path.join(data_dir, 'test.txt')
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        datas = file.readlines()

    if "trec6" in args.data_dir:
        sentence_texts = [sample.split(":")[1].split(" ", 1)[1].strip() for sample in datas]
        raw_labels = [sample.split(":")[0] for sample in datas]
        classes = ['Abbreviation', 'Entity', 'Description and abstract concept',
                   'Human being', 'Location', 'Numeric value']
        label_mapping = {
            'NUM': 'Numeric value',
            'LOC': 'Location',
            'HUM': 'Human being',
            'DESC': 'Description and abstract concept',
            'ENTY': 'Entity',
            'ABBR': 'Abbreviation'
        }
        class_texts = [f"This is a {cls}" for cls in classes]
    elif "sst2" in args.data_dir:
        sentence_texts = [sample.split(maxsplit=1)[1].strip() for sample in datas]
        raw_labels = [int(sample.split(maxsplit=1)[0]) for sample in datas]
        classes = ['positive', 'negative']
        label_mapping = {
            0: 'negative',
            1: 'positive'
        }
        class_texts = [f"The sentence's label is {cls}." for cls in classes]
    # class_texts = [f"{cls}." for cls in classes]
    labels = [label_mapping[label] for label in raw_labels]

    return sentence_texts, class_texts, labels, classes


def data_package(texts, classes, labels):
    data_dict = {
        "sentences": texts,
        "classes": [classes] * len(texts),
        "labels": labels
    }
    dataset = Dataset.from_dict(data_dict)

    return dataset


def collate_fn(batch):
    sentences = [item["sentences"] for item in batch]
    classes = batch[0]["classes"]
    labels = torch.tensor([item["labels"] for item in batch])

    return {"sentences": sentences, "classes": classes, "labels": labels}

