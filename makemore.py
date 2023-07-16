import torch
import matplotlib.pyplot as plt

words = open('/Users/willeickman/Desktop/Sportsbot/names.txt', 'r').read().splitlines()

#Character Set and Count Array
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
N = torch.zeros((27, 27), dtype=torch.int32)

#Bigram Maker
for w in words:
    chs = ['.'] + list(w) +["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

#Visualization
itos = {i:s for s,i in stoi.items()}
plt.figure(figsize=(16,16))
plt.imshow(N, cmap="Blues")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
#plt.show() # displays the vizualtion 

#Probability Matrix
P = (N+1).float()
P /= P.sum(1, keepdim=True)

#Name Generator(Concrete)
g = torch.Generator()
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

#Loss Fuction: Determines Quality of Model
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) +["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
# print(f'LL is {log_likelihood}')
nll = -log_likelihood
# print(f'NLL is {nll}')
print(f'LOSS is {nll/n}')

#Create the Traing Set of Birgrams
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) +["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()


#Initialize the Network
import torch.nn.functional as F
W = torch.randn((27, 27), generator=g, requires_grad=True)

#Gradient Descent
for k in range(200):

    #Forward Pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(num), ys].log().mean() + .001*(W**2).mean()

    #Backward Pass
    W.grad = None
    loss.backward()

    #Update
    W.data += -50 * W.grad
print(loss.item())

#Sample
for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
