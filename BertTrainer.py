from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer

from keras.preprocessing.sequence import pad_sequences

import torch

from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

import numpy as np


def tag_conversion(tags):
    all_tags = sorted(list(set(tags)))
    return [all_tags.index(t) for t in tags]
        

def split(dataframe):
    dataframe.sample(frac=1)
    train_df, val_df, test_df = np.split(dataframe.sample(frac=1), [int(.6*len(dataframe)), int(.8*len(dataframe))])
    train_sentences = [str(s) for s in train_df["text"]]
    train_labels = tag_conversion(train_df["tag"])
    val_sentences = [str(s) for s in val_df["text"]]
    val_labels = tag_conversion(val_df["tag"])
    test_sentences = [str(s) for s in test_df["text"]]
    test_labels = tag_conversion(test_df["tag"])
    
    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels
                                    

def format_data(sentences, labels, tokenizer, maxlen=64, batch_size=32):
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    inputs = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    inputs = pad_sequences(inputs, maxlen=maxlen, dtype="long", truncating="post", padding="post")
    masks = torch.tensor([[float(i>0) for i in t] for t in inputs])

    inputs = torch.tensor(inputs, dtype=torch.long)
    labels = torch.tensor(labels)
                            
    data = TensorDataset(inputs, masks, labels)
    
    return data


class dummy_data_collector():
    def collate_batch(self, features):
        batch = {}
        batch['input_ids'] = torch.stack([f[0] for f in features])
        batch['attention_mask'] = torch.stack([f[1] for f in features])
        batch['labels'] = torch.stack([f[2] for f in features])

        return batch

def train(model, train_dataset, val_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=1,              # total # of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=0, # 500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        data_collator = dummy_data_collector()
    )

    trainer.train()
    trainer.evaluate()
    return trainer.predict(test_dataset)
    
