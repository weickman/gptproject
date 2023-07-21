import torch
import torch.nn as nn
from torch.nn import functional as F

# read it in to inspect it
with open('dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#set of unique characters
chars = sorted(list(text))
vocab_size = len(chars)

#create the mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#encode the text
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 4 #number of sequences
block_size = 8 #max context

torch.manual_seed(1337)
def get_batch(split):
    #generate a batch of inputs(x) and targets(y)
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
xb, yb = get_batch('train')

for b in range (batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx) #(B, T, C)

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B, T)
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        
m = BigramLanguageModel(vocab_size)
out = m(xb, yb)

