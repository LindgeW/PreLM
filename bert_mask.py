from transformers import BertTokenizer, BertForMaskedLM, BertModel
import torch


def load_pretrained_bert(bert_path):
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        bert_model = BertForMaskedLM.from_pretrained(bert_path)
        # bert_model = BertModel.from_pretrained(bert_path)
        bert_model.eval()

    return tokenizer, bert_model


def masked_pred(sent, tokenizer=None, bert_model=None):
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token]
    for i in range(1, len(tokens)-1):
        masked_tokens = tokens[:i] + [tokenizer.mask_token] + tokens[i+1:]
        masked_token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(masked_tokens)])
        segment_ids = torch.tensor([[0]*len(masked_tokens)])
        output = bert_model(masked_token_ids, token_type_ids=segment_ids)
        pred_idx = torch.argmax(output[0][0, i]).item()
        pred_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        print(masked_tokens, ': ', pred_token)


def bert_mlm(sent, pos, tokenizer=None, bert_model=None):
    token_ids = tokenizer.encode(sent)
    mask_id = tokenizer.mask_token_id
    token_ids[pos] = mask_id
    masked_token_ids = torch.tensor([token_ids])
    segment_ids = torch.tensor([[0]*len(masked_token_ids)])
    output = bert_model(masked_token_ids, token_type_ids=segment_ids)
    top_ids = torch.topk(output[0][0, pos], k=10, dim=-1)[1]
    for pred_idx in top_ids:
        pred_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        print(sent, ': ', pred_token)


if __name__ == '__main__':
    tokenizer, bert_model = load_pretrained_bert('bert/en_base')
    sent = 'This story has a good ending .'
    bert_mlm(sent, 4, tokenizer, bert_model)
