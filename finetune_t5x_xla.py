import os
import transformers
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from tqdm import tqdm
from transformers import (T5TokenizerFast, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EncoderDecoderModel, T5ForConditionalGeneration,
                            get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, Dataset


def main():
    NUM_EPOCHS = 7
    max_length = 128
    batch_size = 1
    MAX_SIZE = 100000

    tokenizer = T5TokenizerFast.from_pretrained("checkpoint_4000000")
    model = T5ForConditionalGeneration.from_pretrained("checkpoint_4000000")

    device = xm.xla_device()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('Device: ', device)


    class MyDataset(Dataset):
        def __init__(self, tokenizer, file_source: str, file_target: str):
            self.file_source = file_source
            self.instances = []
            print("Reading corpus: {}".format(file_source))

            # checks
            assert os.path.isfile(file_source)
            f = open(file_source, "r", encoding="utf8")
            g = open(file_target, "r", encoding="utf8")
            lines_source = f.readlines()[:MAX_SIZE]
            lines_target = g.readlines()[:MAX_SIZE]
            for line_s, line_t in tqdm(zip(lines_source, lines_target), total=MAX_SIZE):
                instance = {
                    "source": line_s,
                    "target": line_t
                }
                self.instances.append(instance)

        def __len__(self):
            return len(self.instances)

        def __getitem__(self, i):
            return self.instances[i]


    def my_collate(batch):
        text_batch_source = []
        text_batch_target = []
        for instance in batch:

            text_batch_source.append(instance["source"])
            text_batch_target.append(instance["target"])

        text_batch_source_out = tokenizer(text_batch_source,
                               max_length=max_length, truncation=True, padding="max_length", add_special_tokens=True, return_tensors="pt")
        text_batch_target_out = tokenizer(text_batch_target,
                                          max_length=max_length, truncation=True, padding="max_length", add_special_tokens=True,
                                          return_tensors="pt")

        text_batch_source_out["input_ids"][text_batch_source_out["input_ids"][:, :] == tokenizer.pad_token_id] = -100
        text_batch_source_out["input_ids"][text_batch_source_out["input_ids"][:, :] == tokenizer.pad_token_id] = -100

        return text_batch_source_out, text_batch_target_out



    train_dataset = MyDataset(tokenizer=tokenizer, file_source="./corpus/wiki.txt.train", file_target="./corpus/wiki.txt.train.strip")
    val_dataset = MyDataset(tokenizer=tokenizer, file_source="./corpus/wiki.txt.test", file_target="./corpus/wiki.txt.test.strip")
    test_dataset = MyDataset(tokenizer=tokenizer, file_source="./corpus/wiki.txt.valid", file_target="./corpus/wiki.txt.valid.strip")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
          test_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
          val_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16, collate_fn=my_collate, pin_memory=True, drop_last = True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=16, collate_fn=my_collate, pin_memory=True, drop_last = True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=16, collate_fn=my_collate, pin_memory=True, drop_last = True)


    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Training
    for epoch in range(NUM_EPOCHS):
        xm.master_print(f"Epoch:", epoch)
        para_loader = pl.ParallelLoader(train_dataloader, [device])

        for batch in tqdm(para_loader.per_device_loader(device)):
            model.train()

            batch_source, batch_target = batch[0], batch[1]
            lm_labels = batch_target["input_ids"].to(device)

            input_ids = batch_source["input_ids"].to(device)
            attention_mask_enc = batch_source["attention_mask"].to(device)
            labels = batch_target["input_ids"].to(device)
            attention_mask_dec = batch_target['attention_mask'].to(device)
            optimizer.zero_grad()

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask_enc,
                            labels=labels, 
                            decoder_attention_mask=attention_mask_dec)
            loss = outputs.loss
            xm.master_print("Loss:", loss.item())
            loss.backward()
            optimizer.step()
            xm.mark_step()
            scheduler.step()

    model.save_pretrained("finetuned_t5_diacritics")



# Start training processes
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    main()

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

