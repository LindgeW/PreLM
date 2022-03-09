import torch
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import CrossEntropyLoss
import numpy as np
# https://github.com/Morizeyao/GPT2-Chinese/issues/66


# 官方的gpt-2不支持中文且是BPE分词方式
# 中文的gpt-2模型分词采用的是bert tokenizer的分词方式
# https://gitcode.net/mirrors/Morizeyao/GPT2-Chinese
def load_pretrained_gpt(gpt_path):
    with torch.no_grad():
        # tokenizer = BertTokenizer.from_pretrained(gpt_path)  # zh
        tokenizer = GPT2Tokenizer.from_pretrained(gpt_path)  # en
        model = GPT2LMHeadModel.from_pretrained(gpt_path)
    return model, tokenizer


def gpt_ppl(sent, model, tokenizer):
    model.eval()
    # inputs = tokenizer(sent, padding='max_length', max_length=50, truncation=True, return_tensors="pt")
    inputs = tokenizer(sent, return_tensors='pt')
    bs, sl = inputs['input_ids'].size()
    outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs[1]
    # Shift so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()
    shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).numpy().tolist()
    print('way1:', sent, ppl)
    return ppl


def ppl_score(sentence, model, tokenizer):
    model.eval()
    # tensor_input = tokenizer(sentence, return_tensors='pt').input_ids
    tokens = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    loss = model(tensor_input, labels=tensor_input)[0]
    ppl = np.exp(loss.data.item())
    print(sentence, ppl)
    return ppl


# sents = ["今天是个好日子。",
#         "天今子日。个是好",
#         "这个婴儿有900000克呢。",
#         "我不会忘记和你一起奋斗的时光。",
#         "我不会记忘和你一起奋斗的时光。",
#         "会我记忘和你斗起一奋的时光。"]

sents = ['there is a book on the desk',
         'there is a plane on the desk',
         'there is a book in the desk',
         'there is a apple on the table.',
         'there is an apple in the table.',
         'there is an apple on the table.',
         'I fly to China.',
         'I fly to Sky.',
         ]

model, tokenizer = load_pretrained_gpt('gpt2/en')

for s in sents:
    ppl_score(s, model, tokenizer)