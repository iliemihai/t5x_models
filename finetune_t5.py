import logging, os, sys, json, torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AdamW,
    T5Config,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_linear_schedule_with_warmup
)

model_name = "dumitrescustefan/t5-v1_1-base-romanian"
class TransformerModel (pl.LightningModule):
    def __init__(self, model_name=model_name, lr=0.001, model_max_length=512):
        super().__init__()
        print("Loading T5Model [{}]...".format(model_name))
        self.model_name = model_name
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.config = T5Config.from_pretrained(model_name, num_labels=1, output_hidden_states=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=self.config)

        self.lr = lr
        self.model_max_length = model_max_length

        self.train_y_hat = []
        self.train_y = []
        self.train_loss = []
        self.valid_y_hat = []
        self.valid_y = []
        self.valid_loss = []
        self.test_y_hat = []
        self.test_y = []
        self.test_loss = []


    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )
        return output

    def _step(self, batch):
        batch_source, batch_target = batch[0], batch[1]
        lm_labels = batch_target["input_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch_source["input_ids"],
            attention_mask=batch_source["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch_target['attention_mask']
        )

        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}


    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"test_loss": loss}


    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        return {"avg_test_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)

class MyDataset(Dataset):
    def __init__(self, tokenizer, file_source: str, file_target: str):
        self.file_source = file_source
        self.instances = []
        print("Reading corpus: {}".format(file_source))

        # checks
        assert os.path.isfile(file_source)
        f = open(file_source, "r", encoding="utf8")
        g = open(file_target, "r", encoding="utf8")
        lines_source = f.readlines()
        lines_target = g.readlines()
        for line_s, line_t in zip(lines_source, lines_target):
            instance = {
                "source": line_s,
                "target": line_t
            }
            self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]


tokenizer = AutoTokenizer.from_pretrained(model_name)

def my_collate(batch):
    # batch is a list of batch_size number of instances; each instance is a dict, as given by MyDataset.__getitem__()
    # return is a text_batch, ratings
    # the first two return values are dynamic batching for sentences, and [bs] is the ratings for each of them
    # text_batch is a dict like:
    """
    'input_ids': tensor([[101, 8667, 146, 112, 182, 170, 1423, 5650, 102],
                         [101, 1262, 1330, 5650, 102, 0, 0, 0, 0],
                         [101, 1262, 1103, 1304, 1304, 1314, 1141, 102, 0]]),
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
    """
    text_batch_source = []
    text_batch_target = []
    for instance in batch:

        text_batch_source.append(instance["source"])
        text_batch_target.append(instance["target"])

    text_batch_source_out = tokenizer(text_batch_source,
                           max_length=512, truncation=True, padding="max_length", add_special_tokens=True, return_tensors="pt")
    text_batch_target_out = tokenizer(text_batch_target,
                                      max_length=512, truncation=True, padding="max_length", add_special_tokens=True,
                                      return_tensors="pt")
    return text_batch_source_out, text_batch_target_out

  if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accumulate_grad_batches', type=int, default=64)
    parser.add_argument('--model_name', type=str, default=model_name) #xlm-roberta-base
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--experiment_iterations', type=int, default=1)
    args = parser.parse_args()


    print("Batch size is {}, accumulate grad batches is {}, final batch_size is {}\n".format(args.batch_size, args.accumulate_grad_batches, args.batch_size * args.accumulate_grad_batches))

    model = TransformerModel(model_name=args.model_name, lr=args.lr, model_max_length=args.model_max_length) # need to load for tokenizer
    print("Loading data...")
    train_dataset = MyDataset(tokenizer=tokenizer, file_source="./corpus/wiki.txt.train", file_target="./corpus/wiki.txt.train.strip")
    val_dataset = MyDataset(tokenizer=tokenizer, file_source="./corpus/wiki.txt.test", file_target="./corpus/wiki.txt.test.strip")
    test_dataset = MyDataset(tokenizer=tokenizer, file_source="./corpus/wiki.txt.valid", file_target="./corpus/wiki.txt.valid.strip")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, collate_fn=my_collate, pin_memory=True, drop_last = True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True, drop_last = True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, collate_fn=my_collate, pin_memory=True, drop_last = True)


    print("Train dataset has {} instances.".format(len(train_dataset)))
    print("Valid dataset has {} instances.".format(len(val_dataset)))
    print("Test dataset has {} instances.\n".format(len(test_dataset)))

    itt = 0

    v_p = []
    v_s = []
    v_l = []
    t_p = []
    t_s = []
    t_l = []
    while itt<args.experiment_iterations:
        print("Running experiment {}/{}".format(itt+1, args.experiment_iterations))

        model = TransformerModel(model_name=args.model_name, lr=args.lr, model_max_length=args.model_max_length)

        #from pytorch_lightning.plugins.apex_amp import ApexMixedPrecisionPlugin

        #apex_plugin = ApexMixedPrecisionPlugin(amp_level="O2")
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            #limit_train_batches=5,
            #limit_val_batches=2,
            strategy=DeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=True,
            ),#"deepspeed_stage_3_offload",
            gradient_clip_val=0.5,
            max_epochs=10,
            amp_backend="native",
            #plugins=[apex_plugin],
            precision=32,
            accumulate_grad_batches=args.accumulate_grad_batches,
            #checkpoint_callback=False
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        print("SAVE MDOEL...")
        model.model.save_pretrained('t5_base_diacritics')
        resultd = trainer.test(model, val_dataloader)
        result = trainer.test(model, test_dataloader)
