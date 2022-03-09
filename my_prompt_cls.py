from transformers import BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup
import numpy as np
import torch
import torch.optim as opt
import argparse

# mask_pos = 3   # mask对应的位置

def load_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [line.strip().split('||') for line in f]
    # 模板放在原句之前
    text = ['天气[MASK]，' + x[0] for x in data]
    label = ['天气' + x[1] + '，' + x[0] for x in data]
    data_dict = {'text': text, 'label_text': label}
    return data_dict


def create_model_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    return tokenizer, model


def create_dataset(data_dict, tokenizer, max_len=128):
    def proc_func(example):
        text_token = tokenizer(example['text'], padding=True, truncation=True, max_length=max_len)
        text_token['labels'] = np.array(
            tokenizer(example['label_text'], padding=True, truncation=True, max_length=max_len)["input_ids"])
        return text_token
    # {'input_ids': ..., 'token_type_ids': ..., 'attention_mask': ..., 'labels': ...}
    dataset = proc_func(data_dict)
    return dataset


def batcher(data, batch_size=4, shuffle=False, device=torch.device('cpu')):
    if shuffle:
        ids = np.arange(len(data['labels']))
        np.random.shuffle(ids)
        for k, vs in data.items():  # 同步shuffle
            data[k] = np.asarray(vs)[ids]

    # for k, v in data.items():
    #     if isinstance(v, torch.Tensor):
    #         inputs[k] = v.to(self.args.device)

    for i in range(0, len(data['labels']), batch_size):
        yield {k: torch.tensor(vs[i: i+batch_size], dtype=torch.long).to(device)
               for k, vs in data.items()}


def prompt_train(model, args, train_dataset):
    optimizer = opt.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    nb_batch = (len(train_dataset['labels']) + args.batch_size - 1) // args.batch_size
    total_steps = args.epoch * nb_batch
    schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
    for ep in range(args.epoch):
        nb_correct = 0
        nb_total = 0
        train_loss_sum = 0
        model.train()
        for i, batch in enumerate(batcher(train_dataset, batch_size=args.batch_size, shuffle=True, device=args.device)):
            masked_out = model(**batch, return_dict=True)
            # MLM loss为最终目标，预测被mask的词
            loss = masked_out['loss']
            pred_out = masked_out['logits']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            train_loss_sum += loss.item()
            # gold_lbl = batch['labels'][:, mask_pos].cpu().numpy()
            # pred_lbl = pred_out.data[:, mask_pos].argmax(dim=-1).cpu().numpy()
            mask_ids = (batch['input_ids'] == args.mask_token_id).nonzero(as_tuple=True)
            gold_lbl = batch['labels'][mask_ids].cpu().numpy()
            pred_lbl = pred_out.data[mask_ids].argmax(dim=-1).cpu().numpy()
            nb_correct += (pred_lbl == gold_lbl).sum()
            nb_total += len(gold_lbl)
        acc = nb_correct / nb_total
        print(nb_correct, nb_total)
        print('Epoch: %d, ACC: %.4f' % (ep+1, acc))


def evaluate(model, args, test_dataset):
    print('evaluate ...')
    nb_correct = 0
    nb_total = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(batcher(test_dataset, batch_size=args.batch_size, device=args.device)):
            masked_out = model(**batch, return_dict=True)
            pred_out = masked_out['logits']
            # gold_lbl = batch['labels'][:, mask_pos].cpu().numpy()
            # pred_lbl = pred_out.data[:, mask_pos].argmax(dim=-1).cpu().numpy()  # mask位置取vocab_size维度的向量最大索引
            mask_ids = (batch['input_ids'] == args.mask_token_id).nonzero(as_tuple=True)
            gold_lbl = batch['labels'][mask_ids].cpu().numpy()
            pred_lbl = pred_out.data[mask_ids].argmax(dim=-1).cpu().numpy()
            nb_correct += (pred_lbl == gold_lbl).sum()
            nb_total += len(gold_lbl)
    acc = nb_correct / nb_total
    print(nb_correct, nb_total)
    print('Eval ACC: %.4f' % acc)


def predict(model, tokenizer, args, test_dataset):
    print('predict ...')
    res = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(batcher(test_dataset, batch_size=args.batch_size, device=args.device)):
            masked_out = model(**batch, return_dict=True)
            pred_out = masked_out['logits']
            # pred_lbl = pred_out.data[:, mask_pos].argmax(dim=-1).cpu().numpy()
            mask_ids = (batch['input_ids'] == args.mask_token_id).nonzero(as_tuple=True)
            pred_lbl = pred_out.data[mask_ids].argmax(dim=-1).cpu().numpy()
            # lbl_txt = tokenizer.decode(pred_lbl)
            lbl_txt = tokenizer.convert_ids_to_tokens(pred_lbl)
            res.extend(lbl_txt)
    return res


def set_seeds(seed=1349):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run():
    set_seeds(3347)
    parse = argparse.ArgumentParser('PromptTuning for Text Classification')
    parse.add_argument('--cuda', type=int, default=-1, help='cuda device, default cpu')
    parse.add_argument('-bs', '--batch_size', type=int, default=2, help='batch size')
    parse.add_argument('-ep', '--epoch', type=int, default=20, help='training epoch')
    parse.add_argument('-lr', '--lr', type=float, default=2e-5, help='learning rate')
    parse.add_argument('-ml', '--max_len', type=int, default=128, help='max seq len')
    parse.add_argument('--train_file', type=str, default='train.txt', help='train data file')
    parse.add_argument('--test_file', type=str, default='test.txt', help='test data file')
    args = parse.parse_args()
    args.device = torch.device('cuda', args.cuda) if torch.cuda.is_available() and args.cuda >= 0 else torch.device('cpu')

    tokenizer, model = create_model_tokenizer("bert/zh_base")
    model = model.to(args.device)
    args.mask_token_id = tokenizer.mask_token_id
    print(args.mask_token_id)
    # train
    train_data = load_data(args.train_file)
    train_set = create_dataset(train_data, tokenizer, args.max_len)
    prompt_train(model, args, train_set)
    # evaluate
    test_data = load_data(args.test_file)
    test_set = create_dataset(test_data, tokenizer, args.max_len)
    evaluate(model, args, test_set)
    # inference
    res = predict(model, tokenizer, args, test_set)
    print(res)

# python my_prompt_cls.py --cuda 0 -bs 2 -ep 20 -lr 2e-5 -ml 64 --train_file train.txt --test_file test.txt
run()