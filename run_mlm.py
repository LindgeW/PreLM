# https://github.com/allenai/dont-stop-pretraining/blob/master/scripts/run_language_modeling.py
# https://github.com/lucidrains/mlm-pytorch/blob/master/mlm_pytorch/mlm_pytorch.py
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List
from torch.nn.utils.rnn import pad_sequence

base = "./data/tokenized_files/BC.txt"
pretrain_model = "bert/zh_base"
max_length = 512
epochs = 3
seed = 1357
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class LineByLineTextDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = tokenizer.batch_encode_plus(examples, add_special_tokens=True,
                                                    max_length=max_length, truncation=True)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


class Trainer:
    def __init__(self, model, dataloader, tokenizer, mlm_probability=0.15, lr=5e-5, num_epoch=3, with_cuda=False,
                 cuda_devices=None,
                 log_freq=100):
        if cuda_devices is None:
            cuda_devices = [0]
        if torch.cuda.is_available() and cuda_devices is not None:
            self.device = torch.device("cuda:"+str(cuda_devices[0]))
        else:
            self.device = torch.device('cpu')

        self.lr = lr
        self.model = model
        self.is_parallel = False
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.log_freq = log_freq

        # 多GPU训练
        if with_cuda and cuda_devices is not None and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUS for BERT")
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            self.is_parallel = True

        self.model.train()
        self.model.to(self.device)
        # self.optim = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optim = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=1e-8)
        t_total = len(self.dataloader) * num_epoch
        self.scheduler = get_linear_schedule_with_warmup(self.optim, num_warmup_steps=t_total//10, num_training_steps=t_total)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]) // 10**6)

    def train(self, epoch):
        self.iteration(epoch, self.dataloader)

    def iteration(self, epoch, dataloader, train=True):
        str_code = 'Train'
        total_loss = 0.0
        for i, batch in tqdm(enumerate(dataloader), desc="Training"):
            inputs, labels = self._mask_tokens(batch)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            masked_out = self.model(inputs, labels=labels)  # masked_lm_labels
            lm_loss, output = masked_out['loss'], masked_out['logits']
            loss = lm_loss.mean()

            if train:
                self.model.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                self.scheduler.step()

            total_loss += loss.item()
            post_fix = {
                "iter": i,
                "ave_loss": total_loss / (i + 1)
            }
            if i % self.log_freq == 0:
                print(post_fix)

        print(f"EP{epoch}_{str_code},avg_loss={total_loss / len(dataloader)}")

    def _mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Masked Language Model:
        Prepare masked tokens inputs/labels for mlm: 80% MASK, 10% random, 10% original"""
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.detach().clone()
        # 使用mlm_probability填充张量
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # 获取special token掩码
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        # 将special token位置的概率填充为0
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.pad_token is not None:
            # padding掩码
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            # 将padding位置的概率填充为0
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        # 对token进行mask采样
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # loss只计算masked
        # 80%的概率将masked token替换为[MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10%的概率将masked token替换为随机单词
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # 余下的10%不做改变
        return inputs, labels


def load_data(path):
    examples = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            examples.append(''.join(line.strip().split(' ')))
    return examples


# how to eval: eval_samples -> eval_loss -> ppl
def run_mlm():
    ds = load_data(base)
    tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    model = BertForMaskedLM.from_pretrained(pretrain_model)
    dataset = LineByLineTextDataset(ds, tokenizer, max_length)
    print(" ".join(tokenizer.convert_ids_to_tokens(dataset[5])))

    def collate(examples: List[torch.Tensor]):
        if tokenizer.pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=16, collate_fn=collate)
    trainer = Trainer(model, dataloader, tokenizer, num_epoch=epochs)

    for epoch in range(epochs):
        trainer.train(epoch)

    model.save_pretrained("petrained/")


run_mlm()