import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM


def load_pretrained_bert(bert_path):
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        bert_model = BertForMaskedLM.from_pretrained(bert_path)
    return bert_model, tokenizer


# 对于给定的sentence，按顺序依次mask掉一个token，并计算所预测单词的nll loss，将所有的token的loss求和再取平均，最后取以自然数为底的次方即为该句话的PPL
def bert_ppl(sent, model, tokenizer):
    model.eval()
    tokenize_input = tokenizer.tokenize(sent)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    sen_len = len(tokenize_input)
    sent_loss = 0.
    for i, word in enumerate(tokenize_input):
        # add mask to i-th character of the sentence
        tokenize_input[i] = tokenizer.mask_token
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        output = model(mask_input)
        pred_scores = output[0]
        ps = torch.log_softmax(pred_scores[0, i], dim=0)
        word_loss = ps[tensor_input[0, i]]
        sent_loss += word_loss.item()
        tokenize_input[i] = word   # restore
    ppl = np.exp(-sent_loss / sen_len)
    print(sent, ppl)
    return ppl


def tensor_bert_ppl(sent, model, tokenizer):
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    tensor_input = tokenizer.encode(sent, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, mask_token_id)
    labels = repeat_input.masked_fill(masked_input != mask_token_id, -100)
    loss, _ = model(masked_input, labels=labels)
    res = np.exp(loss.item())
    print(sent, res)
    return res


# sents = ['我来自北京',
#          '我出自于北京',
#          '我来自火星',
#          '北京雾霾很严重',
#          '北京雾霾很严峻',
#          '我将飞往日本东京',
#          '我将飞往法国东京',
#          '周杰伦在哈尔滨篮球公园跑步',
#          '小明在上海城市博物馆体锻']

sents = ["今天是个好日子。",
        "天今子日。个是好",
        "这个婴儿有900000克呢。",
        "我不会忘记和你一起奋斗的时光。",
        "我不会记忘和你一起奋斗的时光。",
        "会我记忘和你斗起一奋的时光。",
         "天下没有免费的午餐",
         "天下没有免费的晚餐"]

# sents = ['there is a apple on the table .',
#          'there is an apple in the table .',
#          'there is an apple on the table .']

model, tokenizer = load_pretrained_bert('bert/zh_base')

for s in sents:
    bert_ppl(s, model, tokenizer)
    # tensor_bert_ppl(s, model, tokenizer)
