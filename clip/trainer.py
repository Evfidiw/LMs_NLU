import os
import logging
import pdb

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from sklearn.metrics import f1_score, accuracy_score

from dataset import data_processing, data_package, collate_fn
from model import MyModel


class MyTrainer():
    def __init__(self, args, device, processor):
        self.args = args
        self.model = MyModel(args, device)
        self.processor = processor
        """
        self.    train_sentence, train_class, train_label = data_processing("train", args)
        

        """

    def set_logger(self, output_dir, name):
        log_file_path = os.path.join(output_dir, "training.log")
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                               datefmt='%m/%d/%Y %H:%M:%S')
            file_handler.setFormatter(file_formatter)

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_formatter = logging.Formatter('%(message)s')
            stream_handler.setFormatter(stream_formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
        return logger

    def compute_metrics(self, p):
        preds = p.predictions.argmax(-1)
        return {
            'accuracy': accuracy_score(p.label_ids, preds),
            'f1': f1_score(p.label_ids, preds, average='weighted')
        }

    def training(self, args, device):
        print('--------------- begin training ---------------')
        logger = self.set_logger(args.logging_dir, 'my_logger')
        logger.info("dataset: {}".format(args.data_dir))
        logger.info("model: {}".format(args.model_name))

        train_texts, train_classes, train_labels, class_names = data_processing("train", args)
        train_labels = torch.tensor([class_names.index(label) for label in train_labels]).to(device)
        train_dataset = data_package(train_texts, train_classes, train_labels)

        dev_dataset = None
        if "sst2" in args.data_dir:
            dev_texts, dev_classes, dev_labels, _ = data_processing("dev", args)
            dev_labels = torch.tensor([class_names.index(label) for label in dev_labels]).to(device)
            dev_dataset = data_package(dev_texts, dev_classes, dev_labels)

        if dev_dataset is not None:
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                evaluation_strategy="epoch",  # "no", "steps", "epoch"
                learning_rate=args.learning_rate,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.train_bsize,
                save_strategy="epoch",
                load_best_model_at_end=True,
                # metric_for_best_model="eval_accuracy",
                weight_decay=0,
                logging_dir=args.logging_dir,
                logging_strategy="epoch",
                seed=args.seed,
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                data_collator=collate_fn,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
            )
        else:
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                evaluation_strategy="no",  # "no", "steps", "epoch"
                learning_rate=args.learning_rate,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.train_bsize,
                save_strategy="no",
                weight_decay=0,
                logging_dir=args.logging_dir,
                logging_strategy="epoch",
                seed=args.seed,
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=collate_fn,
                compute_metrics=self.compute_metrics,
            )

        trainer.train()

        if args.save_model:
            trainer.save_model(args.output_dir)
        for obj in trainer.state.log_history:
            logger.info(str(obj))

    def testing(self, args, device):
        print('--------------- begin testing! ---------------')
        test_texts, test_classes, test_labels, class_names = data_processing("test", args)
        test_labels = torch.tensor([class_names.index(label) for label in test_labels])

        model = self.model
        model.eval()
        with torch.no_grad():
            logits = model(test_texts, test_classes)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            # print("Predictions:", predictions)

        acc = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
        print(f"Accuracy: {acc}, F1 Score: {f1}")

    def train(self, args, device):
        if args.train:
            self.training(args, device)
        if args.test:
            self.testing(args, device)
        if not args.train and not args.test:
            print("No action specified. Please use --train and/or --test.")
