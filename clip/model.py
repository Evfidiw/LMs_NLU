import pdb

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class MyModel(nn.Module):
    def __init__(self, args, device):
        super(MyModel, self).__init__()
        model_name = args.model_name
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, sentences, classes, labels=None):
        """ How to forward """
        class_inputs = self.processor(text=classes, return_tensors="pt", padding=True).to(self.device)
        text_inputs = self.processor(text=sentences, return_tensors="pt", padding=True).to(self.device)
        class_features = self.model.get_text_features(**class_inputs).to(self.device)
        text_features = self.model.get_text_features(**text_inputs).to(self.device)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        logits= torch.matmul(text_features, class_features.T)

        """ Calculate Loss """
        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)
        return (loss, logits) if loss is not None else logits
