import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import TensorDataset, DataLoader
# reference: \transformers\generation_utils.py

def select_greedy(logits):
    next_token_logits = logits[:, -1, :]
    # Greedy decoding
    next_token = torch.argmax(next_token_logits, dim=-1)
    return next_token


def select_topk(logits, k=10):
    # next_token = random.choice(logits[0, -1, :].sort(descending=True)[1][:k]).item()
    next_token_logits = logits[:, -1, :]
    top_k = min(max(k, 1), next_token_logits.size(-1))
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
    next_token_logits[indices_to_remove] = -float("Inf")
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    # multinominal方法可以根据给定权重对数组进行多次采样，返回采样后的元素下标
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_token


def select_topp(logits, p=0.75):
    next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cum_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    next_token_logits[indices_to_remove] = -float("Inf")
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    # multinominal方法可以根据给定权重对数组进行多次采样，返回采样后的元素下标
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_token


def read_data(path='./romeo_and_juliet.txt'):
    with open(path, 'r', encoding='utf-8') as fin:
        ds = fin.read()
    return ds


def data_processor(dataset, tokenizer, max_len=32):
    indexed_text = tokenizer.encode(dataset)
    ds_cut = []
    for i in range(0, len(indexed_text)-max_len, max_len):
        # 将串切成长度为max_len
        ds_cut.append(indexed_text[i: i+max_len])

    ds_tensor = torch.tensor(ds_cut)
    train_set = TensorDataset(ds_tensor, ds_tensor)  # 数据和标签相同
    return DataLoader(dataset=train_set, batch_size=8, shuffle=False)


def train(train_loader, model, ep=30, device=torch.device('cpu')):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, eps=1e-8)
    print(next(model.parameters()).device)
    model.train()
    model.to(device)
    for i in range(ep):
        total_loss = 0.
        for bi, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss, logits, _ = model(data, labels=target)
            print('loss:', loss.data.item())
            total_loss += loss
            loss.backward()
            optimizer.step()
        print('train loss:', total_loss / len(train_loader))
    return model


def inference(model, tokenizer, prefix=None, max_len=100, top_k=20, top_p=0.75, temperature=1.):
    print('inference ... ')
    print(next(model.parameters()).device)
    model.eval()
    indexed_tokens = tokenizer.encode(prefix)
    tokens_tensor = torch.tensor([indexed_tokens])
    final_pred_text = prefix
    cur_len = tokens_tensor.size(-1)
    for _ in range(max_len):
        with torch.no_grad():
            output = model(tokens_tensor)
            logits = output[0]  # (batch_size, cur_len, vocab_size)

        if temperature != 1:
            logits /= temperature

        next_idx = select_topk(logits, k=top_k)
        # next_idx = select_topp(logits, p=0.75)

        final_pred_text += tokenizer.decode(next_idx)
        if tokenizer.eos_token in final_pred_text:
            break

        # indexed_tokens += [next_idx]
        # tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = torch.cat([tokens_tensor, next_idx.unsqueeze(-1)], dim=-1)
        cur_len += 1
    print(cur_len)
    return final_pred_text


tokenizer = GPT2Tokenizer.from_pretrained('gpt2/en')
model = GPT2LMHeadModel.from_pretrained('gpt2/en')
# ds = read_data('./romeo_and_juliet.txt')
# train_loader = data_processor(ds, tokenizer)
# model = train(train_loader, model, ep=3, device=torch.device('cuda', 0))
pred_text = inference(model.to('cpu'), tokenizer,
                      'Yesterday, Jack said he saw an alien,',
                      top_k=20,
                      top_p=0.8,
                      temperature=0.5)
print(pred_text)

