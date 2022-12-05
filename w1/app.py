# %% [markdown]
# <a href="https://colab.research.google.com/github/starship006/ARENA-work/blob/main/w1/w1d4.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Training Shakespeare Himself

# %% [markdown]
# For those who are not part of the ARENA program and are curious as to what this is, this was my first significant AI/ML project! I made components for a decoder-only transformer, and trained it on a corpus consisting of text from Shakespeare. Scroll to the bottom to see some output :)

# %%
!pip install fancy_einsum einops

# %%
import torch as t
import numpy as np
from torch import nn
import fancy_einsum as einsum
import einops
import pandas as pd


# %% [markdown]
# ## transformer functions
# 
# 

# %% [markdown]
# This will be from the transformer components I made earlier this week, but I'll put put down optimizations so it can use the GPU.
# 
# And I did just that. The speed improvements are MASSIVE, wow!

# %%
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
t.cuda.is_available()

# %%
def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int):
    '''
    Implements multihead masked attention on the matrices Q, K and V.

    Q: shape (batch, seq_len, nheads*headsize)
    K: shape (batch, seq_len, nheads*headsize)
    V: shape (batch, seq_len, nheads*headsize)
    '''
    
    Q = einops.rearrange(Q, 'b s (n h) -> b n s h', n = num_heads)
    K = einops.rearrange(K, 'b s (n h) -> b n s h', n = num_heads)
    V = einops.rearrange(V, 'b s (n h) -> b n s h', n = num_heads)


    scores = einsum.einsum('b n k h, b n s h -> b n s k', K, Q)
    assert scores.shape == t.Size([Q.shape[0], num_heads,Q.shape[2], K.shape[2]])

    scores = scores / np.sqrt(Q.shape[-1])
    attention = scores + t.triu(t.ones_like(scores,device = device) * float("-inf"), diagonal=1) # THIS IS STOLEN FROM JAY - testing it out
    softed = t.softmax(attention,dim=-1)
    result =  einsum.einsum('batch numheads seqQ seqK, batch numheads seqK headsize -> batch numheads seqQ headsize',softed, V)
    return einops.rearrange(result, 'batch numheads seqQ headsize -> batch seqQ (numheads headsize)')

# %%
class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.WQKV = t.nn.Linear(self.hidden_size, 3 * hidden_size) # TODO: why do we use a linear layer here? aren't they matricies?
        self.W0 = t.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        #print("YO?")
        x = x.float() # seems like it needs to be a float!
        QKV = self.WQKV(x)
        Q = QKV[:,:,:self.hidden_size]
        K = QKV[:,:,self.hidden_size:self.hidden_size * 2]
        V = QKV[:,:,self.hidden_size * 2:]
        assert Q.shape == K.shape == V.shape == x.shape
        return self.W0(multihead_masked_attention(Q,K,V,self.num_heads))

# %%
from dataclasses import dataclass

@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05

# %%
# from yesterday
class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim: int, max_seq_len: int = 5000):
        super().__init__()
        self.dim = embedding_dim
        self.length = max_seq_len

        # mostly copied. i understand this, just need to work on 
        # making more tensors and getting more exposure to methods of making tensors
        def P (delta):
            n = 10000 # hardcoded
            d = embedding_dim
            l = max_seq_len
            sin_array = np.sin(delta / n ** (2 * np.arange(d//2) / d))
            cos_array = np.cos(delta / n ** (2 * np.arange(d//2) / d))

            array = np.zeros(d)
            array[::2] = sin_array
            array[1::2] = cos_array

            return array

        tokenArray = []
        for i in range(max_seq_len):
            tokenArray.append(P(i)) # changed from previous design
        
        self.multMax = t.tensor(np.array(tokenArray), dtype=t.float, device = device)
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq_len, embedding_dim)
        '''
        return x + self.multMax[:x.shape[1]]


# %%
class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(config.dropout)
        )
    def forward(self, x: t.Tensor):
        x = x.float() # seems like it needs to be a float!
        return self.layers(x).float() # ima do the same thing again!


# %%
class DecoderBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attentionBlock = nn.Sequential(
            MultiheadMaskedAttention(config.hidden_size,  config.num_heads),
            nn.LayerNorm(config.hidden_size)
        )
        self.MLP = nn.Sequential(
            MLP(config),
            nn.LayerNorm(config.hidden_size)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        partOne = x + self.attentionBlock(x)
        return (partOne + self.MLP(partOne)).float() # seems like it needs to be a float!
        

# %%
class DecoderOnlyTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.tokenize = nn.Embedding(config.vocab_size, config.hidden_size).to(device)
        self.positionize = PositionalEncoding(config.hidden_size,config.max_seq_len)
        self.restModel = nn.Sequential(
            nn.Dropout(config.dropout),
            *[DecoderBlock(config) for i in range(config.num_layers)],
            nn.LayerNorm(config.hidden_size),
        )
        self.unembed = self.tokenize.weight.T.to(device)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.tokenize(x)
        x = self.positionize(x)
        toUnembed = self.restModel(x).to(device)
        return toUnembed@self.unembed

# %% [markdown]
# ## Data Prep

# %% [markdown]
# Make the dataset to parse through all of the words

# %%
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

class CustomTextDataset(Dataset):
    def __init__(self, words, seq_len, fractionOfWords):
        self.fractionOfWords = fractionOfWords
        self.words = words
        self.setOfWords = set(words)
        self.seq_len = seq_len
        self.max_len = len(self.words) - (self.seq_len + 1)
        self.vocab_size = len(self.setOfWords)
        self.word_to_token = {word: idx for (idx, word) in enumerate(sorted(self.setOfWords))}
        self.token_to_word = {idx: word for (idx, word) in enumerate(sorted(self.setOfWords))}
        self.allTokens = t.tensor([self.word_to_token[word] for word in self.words],device = device)
        
        if (self.fractionOfWords > 0.9):
            print("Probably don't do this. Errors may about")

    def __len__(self):
        return int(self.max_len * self.fractionOfWords)

    def __getitem__(self, idx):
        tokens = self.allTokens[idx:idx + self.seq_len + 1]
        input = tokens[:-1]
        target = tokens[1:]
        return input, target 

    def getDataSize(self):
        return self.vocab_size

    def convertToTokens(self, phrase: list) -> t.tensor:
        return t.tensor([self.word_to_token[word] for word in phrase],device = device)

    def convertStringToTokenList(self, phrase: str) -> list:
        words = re.split(r"\b", phrase)
        return [self.word_to_token[word] for word in words]

    def convertToText(self, tokens: t.tensor):
        temp = []
        for i, value in enumerate(tokens):
            #print(value.item())
            temp.append(self.token_to_word[value.item()])
        return temp

    def decodeList(self, words: list):
        temp = []
        for value in words:
            temp.append(self.token_to_word[value])
        return temp
        
    def listToString(self, words: list) -> str:
        temp = ""
        for word in words:
            temp = temp + word
        return temp

# %%
file = open("shakespeare.txt")
text = file.read()
words = re.split(r"\b", text)

fractionOfWords = 0.1 # what percent of the corpus to train on 


lengthOfSeq = 100

shak = CustomTextDataset(words, lengthOfSeq, fractionOfWords)

# %% [markdown]
# ## Running this data through a transformer

# %%
trainloader = DataLoader(shak, batch_size=32,shuffle=True)

# this specific one trained for 24 minutes and 9 seconds on colab GPU

thisConfig = TransformerConfig(
    num_layers = 4, # 6 layers in the Attention paper
    num_heads = 4, # 8 heads in Attention paper
    vocab_size = trainloader.dataset.getDataSize(), # 37000 tokens in Attention paper (?)
    hidden_size = 512, # recall that this = num_heads * headsize | 512 is the embedding dim used in Attention paper
    max_seq_len = lengthOfSeq, 
    dropout = 0.1, # same as Attention paper
    layer_norm_epsilon=0.00001
)




# %%
use_pretrained = True
if use_pretrained:
    print("Using Pre-trained Model!")
    myTransformer = DecoderOnlyTransformer(thisConfig).to(device)
    optimizer = t.optim.Adam(myTransformer.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss().to(device)
    myTransformer.load_state_dict(t.load("toInfer.pt", map_location=device))
    myTransformer.eval()
else:
    print("Training Model... better hope you got enough GPU!")
    myTransformer = DecoderOnlyTransformer(thisConfig).to(device)
    optimizer = t.optim.Adam(myTransformer.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss().to(device)
    NUM_EPOCHS = 1

    losses = []
    myTransformer.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        for inputs, targets in trainloader:
            outputs = myTransformer(inputs).to(device)
            targets = t.nn.functional.one_hot(targets, num_classes=trainloader.dataset.getDataSize()).float().to(device)
            
            outputs = einops.rearrange(outputs, 'batch seq vocab -> (batch seq) vocab')
            targets = einops.rearrange(targets, 'batch seq vocab -> (batch seq) vocab')

            outputs = outputs.to(device)
            targets = targets.to(device)
            loss = criterion(outputs,targets).to(device)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


# %%
if not use_pretrained:
    df = pd.DataFrame(losses)
    df.plot()

# %%
# quick test - use the sample method if you wish to actually use the transformer: 

myTransformer.eval()

testPhrase = ["Be", " ", "not", " ", "afraid", " ", "to", " ", "the", " ", "Florentine", "\n",
              "And"]
input = shak.convertToTokens(testPhrase)
input = input[None, :]
tokens = myTransformer(input).argmax(dim=-1)[0]
print(tokens)
shak.convertToText(tokens)

# %% [markdown]
# # Sampling

# %%
def apply_sampling_methods(input_ids: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0) -> int:
  # returns a next token based on provided sampling method
  # thanks callum for the this method
  assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
  assert temperature >= 0, "Temperature should be non-negative"
  assert 0 <= top_p <= 1.0, "Top-p must be a probability"
  assert 0 <= top_k, "Top-k must be non-negative"
  assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

  if temperature == 0:
    return greedy_search(logits)
  if temperature != 1.0:
    logits = apply_temperature(logits, temperature)
  if freq_penalty != 0.0:
    logits = apply_freq_penalty(input_ids, logits, freq_penalty)
  if top_k > 0:
    return sample_top_k(logits, top_k)
  if top_p > 0:
    return sample_top_p(logits, top_p)
  return sample_basic(logits)


def sample_tokens(
    model,
    encodeMethod,
    decodeMethod,
    initial_text: str,
    max_tokens_generated = 40,
    **kwargs) -> list:
    # samples tokens until model outputs eos_token_id or token limit reached

    

    

    model.eval()
    input_ids: list = encodeMethod(initial_text)
    generated_ids = []
    device = next(model.parameters()).device #what is next doing here?

    tokens_to_generate = max_tokens_generated - len(input_ids)
    for _ in range(tokens_to_generate):
        #print(input_ids + generated_ids)
        new_input_ids = t.tensor(input_ids + generated_ids, dtype=t.int64, device=device)
        #print(new_input_ids.unsqueeze(0).shape)
        logits = model(new_input_ids.unsqueeze(0))[0, -1]
        #print(logits.shape)
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated_ids.append(new_token)

      
    return decodeMethod(input_ids + generated_ids)


# quick test:

myTransformer.eval()

testPhrase = ["Be", " ", "not", " ", "afraid", " ", "to", " ", "the", " ", "Florentine", "\n",
              "And"]
input = shak.convertToTokens(testPhrase)
type(input)


# %%
def greedy_search(logits):
    '''
    returns the most likely next token, BUT THE TIEBREAKER IS INCORRECT!
    i got lazy - it *is* deterministic, but it just doesn't necessarily
    choose the smallest word out of the tie. perhaps treat it as a symbol
    of my ingenuity?
    '''
    return logits.argmax(dim=-1).item()

# %%
def sample_basic(logits) -> int:
    '''
    samples from the distributions, possibly with temp and freq changes applied

    logits: shape (vocab_size, ) - unnormalized log-probabilities

    return: a sampled token
    '''
    probs = t.distributions.categorical.Categorical(logits=logits)
    return probs.sample().item()

N = 20000
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, probs, atol=0.01, rtol=0)
print("Tests passed!")

# %%
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    (vocab_size,) = logits.shape
    id_freqs = t.bincount(input_ids, minlength=vocab_size)
    return logits - freq_penalty * id_freqs

bieber_prompt = "And I was like baby, baby, baby, oh Like, baby, baby, baby, no Like, baby, baby, baby, oh I thought you'd always be mine, mine"
input_ids = shak.convertStringToTokenList(bieber_prompt)
logits = t.ones(shak.getDataSize()).to(device)
penalized_logits = apply_freq_penalty(t.tensor(input_ids).to(device), logits, 2.0)
#i believe mine is different!
#assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
#assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"
#print("Tests passed!")

print(penalized_logits[2037].item()) # should be low since it was found!
shak.convertStringToTokenList("And")

# %%
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    assert temperature > 0, "temp cannot be less than or equal to 0"

    return logits / temperature

logits = t.tensor([1, 2]).log()
cold_logits = apply_temperature(logits, 0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)
hot_logits = apply_temperature(logits, 1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)
print("Tests passed!")

# %%
N_RUNS = 1
your_prompt = "We are the champions, my friends"
cases = [
    ("High freq penalty", dict(freq_penalty=100.0)),
    ("Negative freq penalty", dict(freq_penalty=-1.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]
for (name, kwargs) in cases:
    for i in range(N_RUNS):
        output = sample_tokens(myTransformer, shak.convertStringToTokenList,shak.decodeList, your_prompt, max_tokens_generated=24, **kwargs)
        print(f"Sample {i} with: {name} ({kwargs}):")
        print(f"Your model said: {shak.listToString(output)}\n")

# %%
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    topk = t.topk(logits,top_k).indices
    almost_zeroes = t.ones(logits.shape) * t.inf * -1
    for _, token in enumerate(topk):
        almost_zeroes[token] = 0
    logits = logits + almost_zeroes
    return sample_basic(logits)

k = 3
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
expected = probs.clone()
expected[:-k] = 0
expected /= expected.sum()
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, expected, atol=0.01, rtol=0)
print("Tests passed!")

# %%
def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    # find the indices of importang logits
    sorted, indices = t.sort(logits,descending=True)
    probs = t.nn.functional.softmax(sorted, dim=-1)
    num_words_kept = 0
    sum = 0
    while sum < top_p:
        sum = sum + probs[num_words_kept]
        num_words_kept = num_words_kept + 1
        

    if num_words_kept < min_tokens_to_keep:
        num_words_kept = min_tokens_to_keep
    
    important_indices = indices[:num_words_kept]

    # prepare tensor to zero out small logits
    almost_zeroes = t.ones(logits.shape) * t.inf * -1
    for _, token in enumerate(important_indices):
        almost_zeroes[token] = 0
    logits = logits + almost_zeroes
    return sample_basic(logits)

N = 2000
unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
print("top_p of 0.5 or lower should only return token 2: ", counts)
assert counts[0] == 0 and counts[1] == 0

N = 2000
unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
print("top_p in (0.5, 0.8] should return tokens 1 and 2: ", counts)
assert counts[0] == 0

N = 4000
top_p = 0.71
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
expected = probs.clone()
expected[0:2] = 0
expected /= expected.sum()
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, expected, atol=0.01, rtol=0.0)

print("All tests passed!")

# %% [markdown]
# # Speak, Shakespeare!

# %%
input = "Death waits at the door"

print(shak.listToString(sample_tokens(myTransformer,shak.convertStringToTokenList,shak.decodeList,
                                     input, 80,freq_penalty=0.1, top_k = 10)))

# %% [markdown]
# ## Save the model for future use
# (This was over 20 minutes of GPU computation. Not too shabby!)

# %%
t.save(myTransformer.state_dict(), "toInfer.pt")

# %% [markdown]
# # Publish to Gradio
# About a month after making this I realized this should be online. I'll push this to gradio

# %%
import gradio as gr
def speak(input, tokenLength):
    return shak.listToString(sample_tokens(myTransformer,shak.convertStringToTokenList,shak.decodeList,
                                    input, tokenLength,freq_penalty=0.1, top_k = 10))


model = gr.Interface(fn=speak,
                    inputs=["text", gr.Slider(40, 80, step=1)],
                    outputs="text",
                    title = "speak shakespeare, speak!")

model.launch(share=True) 

# %%



