# -*- coding: utf-8 -*-
import sys, os, getopt
from transformers import Trainer, TrainingArguments, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import BartTokenizer, AutoModelWithLMHead, BertTokenizer
from transformers import BartForConditionalGeneration

config = {"model": "bert-base-cased",
          "tokenizer":BertTokenizer,
          "out":"./"}


class dummy_data_collector():
    def collate_batch(self, features):
        batch = {}
        batch['input_ids'] = torch.stack([f[0] for f in features])
        batch['attention_mask'] = torch.stack([f[1] for f in features])
        batch['labels'] = torch.stack([f[2] for f in features])

        return batch

def train_lm(input_file, output_model):
    tokenizer = config["tokenizer"].from_pretrained(config["model"])
    model = AutoModelWithLMHead.from_pretrained(config["model"])

    
    dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=input_file, block_size=32)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True)

    training_args = TrainingArguments(
        output_dir=config["out"] + output_model,          # output directory
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=.0001,               # strength of weight decay
        learning_rate=1e-4,
        logging_dir='./logs',            # directory for storing logs
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        do_save_full_model=True
    )

    trainer.train()
    trainer.save_model()
                                        


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        opts, args = getopt.getopt(argv[1:], "hr:m:dl:vc", ["help", "run-type", "max", "dates", "language", "valid", "covid"])
    except Exception as e:
        print ("Error in args : " + str(argv[1:]))
        print (e)
        return 2
    
    for o in opts:
        if o[0] == "-m":
            config["data_max"] = int(o[1])

    for a in args:
        train_lm(a, "test")
        

if __name__ == "__main__":
    sys.exit(main())
