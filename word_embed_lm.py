import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
torch.manual_seed(1)
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
N_EPHCNS = 100


def eng_word_tokenize(sent, keep_sep=True):
    if keep_sep:
        sep = r"([,.!:;-?()<>{}|\s+])"   # 保留分隔符
    else:
        sep = r"[,.!:;-?()<>{}|\s]+"    # 不保留分隔符
    tokens = re.split(sep, sent.strip())
    tokens = [t for t in tokens if t != '' and t is not None]
    return tokens


test_sentence = """
Word embeddings are dense vectors of real numbers,
one per word in your vocabulary. In NLP, it is almost always the case
that your features are words! But how should you represent a word in a
computer? You could store its ascii character representation, but that
only tells you what the word is, it doesn’t say much about what it means
(you might be able to derive its part of speech from its affixes, or properties
from its capitalization, but not much). Even more, in what sense could you combine
these representations?"""

test_sentence = eng_word_tokenize(test_sentence)


# 三元模型语料准备
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_ix.items()}


class NGramLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        logits = self.linear2(out)
        return logits


loss_function = nn.CrossEntropyLoss()
model = NGramLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练语言模型
for epoch in range(N_EPHCNS):
    total_loss = 0.
    for context, target in trigrams:
        context_var = torch.LongTensor([word_to_ix[w] for w in context])
        model.zero_grad()
        logits = model(context_var)
        loss = loss_function(logits, torch.LongTensor([word_to_ix[target]]))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    print('\r epoch[{}] - loss: {:.6f}'.format(epoch, total_loss))

word, label = trigrams[2]
word = torch.LongTensor([word_to_ix[i] for i in word])
out = model(word)
predict_label = out.argmax(dim=-1)
predict_word = idx_to_word[predict_label.data[0].item()]
print('real word is {}, predict word is {}'.format(label, predict_word))
