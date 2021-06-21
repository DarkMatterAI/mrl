import sys
sys.path.append('../')

import torch
torch.cuda.set_device(1)

from fastai.text.all import *
from fastai.vision.all import *

from mrl.imports import *
from mrl.core import *
from mrl.chem import *
from mrl.templates import *
from mrl.torch_imports import *
from mrl.torch_core import *
from mrl.layers import *
from mrl.dataloaders import *
from mrl.g_models import *
from mrl.agent import *
from mrl.policy_gradient import PolicyGradient, TRPO, PPO
from mrl.environment import *

df = pd.read_csv('../../smiles_datasets/shard_0.csv', usecols=['smiles'])

df = df[df.smiles.map(lambda x: 8<len(x)<=100)]
df = df[df.smiles.map(lambda x: not '.' in x)]
vocab = CharacterVocab(SMILES_CHAR_VOCAB)
print(df.shape)

bs = 500

train_df = df.sample(frac=0.99, random_state=42)
valid_df = df[~(df.index.isin(train_df.index))]

train_ds = Vec_Recon_Dataset(train_df.smiles.values, vocab, ECFP6)
valid_ds = Vec_Recon_Dataset(valid_df.smiles.values, vocab, ECFP6)

train_dl = train_ds.dataloader(bs, shuffle=True, num_workers=10)
valid_dl = valid_ds.dataloader(bs, shuffle=False, num_workers=10)

dl = DataLoaders(train_dl, valid_dl)

# encoder = MLP_Encoder(2048, [1024, 512], 512, [0.1, 0.1])
# d_vocab = len(vocab.itos)
# d_embedding = 256
# d_hidden = 1024
# d_latent = 512
# n_layers = 3
# lstm_drop = 0.0
# lin_drop = 0.0
# bidir = False
# condition_hidden = True
# condition_output = False
# bos_idx = vocab.stoi['bos']

encoder = MLP_Encoder(2048, [1024, 512, 512, 512], 512, [0.1, 0.1, 0.1, 0.1])
d_vocab = len(vocab.itos)
d_embedding = 400
d_hidden = 1552
d_latent = 512
n_layers = 6
lstm_drop = 0.0
lin_drop = 0.0
bidir = False
condition_hidden = True
condition_output = False
bos_idx = vocab.stoi['bos']

model = Conditional_LSTM_LM(encoder, d_vocab, d_embedding, d_hidden, d_latent, n_layers,
                lstm_drop, lin_drop, bidir, condition_hidden, condition_output, bos_idx)

class FPCallback(Callback):
    "Move data to CUDA device"
    def before_batch(self): 
        self.learn.xb = self.learn.xb[0]

learn = Learner(dl, model, loss_func=CrossEntropyLossFlat(), cbs=[FPCallback()])

learn = learn.to_fp16()

learn.fit_one_cycle(1, 1e-3)

torch.save(learn.model.cpu().state_dict(), '../nbs/untracked_files/fp_cond_lstm_lm_zinc_large.pt')


# train_ds = TextDataset(train_df.smiles.values, vocab)
# valid_ds = TextDataset(valid_df.smiles.values, vocab)

# train_dl = train_ds.dataloader(bs, shuffle=True, num_workers=0)
# valid_dl = valid_ds.dataloader(bs, shuffle=False, num_workers=0)

# dl = DataLoaders(train_dl, valid_dl)

# gc.collect()

# d_vocab = len(vocab.itos)
# d_embedding = 256
# d_hidden = 1024
# n_layers = 3
# lstm_drop = 0.
# lin_drop = 0.
# bos_idx = vocab.stoi['bos']
# bidir = False
# tie_weights = True

# model = LSTM_LM(d_vocab, d_embedding, d_hidden, n_layers,
#                 lstm_drop, lin_drop, bos_idx, bidir, tie_weights)



# learn = Learner(dl, model, loss_func=CrossEntropyLossFlat())

# learn = learn.to_fp16()

# learn.fit_one_cycle(1, 1e-3)

# torch.save(learn.model.cpu().state_dict(), '../nbs/untracked_files/lstm_lm_zinc.pt')


# print('loading data')
# df = pd.read_csv('../../smiles_datasets/shard_0.csv', usecols=['smiles'])
# df = df[df.smiles.map(lambda x: 8<=len(x)<=90)]
# print(df.shape)
# df2 = pd.read_csv('../../smiles_datasets/chembl/chembl.csv', usecols=['smiles', 'smiles_nc'])
# df2 = df2[df2.smiles.map(lambda x: 8<=len(x)<=90)]
# print(df2.shape)

# df3 = pd.read_csv('../../smiles_datasets/shard_1.csv', usecols=['smiles'])
# df3 = df3[df3.smiles.map(lambda x: 8<=len(x)<=90)]
# print(df3.shape)

# df = pd.concat([df['smiles'], df2['smiles'], df3['smiles']])
# print(df.shape)

# vocab = CharacterVocab(SMILES_CHAR_VOCAB)

# bs = 600

# train_df = df.sample(frac=0.99, random_state=42)
# valid_df = df[~(df.index.isin(train_df.index))]

# train_ds = TextDataset(train_df.values, vocab)
# valid_ds = TextDataset(valid_df.values, vocab)

# train_dl = train_ds.dataloader(bs, shuffle=True, num_workers=0)
# valid_dl = valid_ds.dataloader(bs, shuffle=False, num_workers=0)

# dl = DataLoaders(train_dl, valid_dl)


# gc.collect()

# # d_vocab = len(vocab.itos)
# # d_embedding = 256
# # d_hidden = 1024
# # n_layers = 3
# # lstm_drop = 0.
# # lin_drop = 0.
# # bos_idx = vocab.stoi['bos']
# # bidir = False
# # tie_weights = True


# d_vocab = len(vocab.itos)
# d_embedding = 400
# d_hidden = 1552
# n_layers = 4
# lstm_drop = 0.
# lin_drop = 0.
# bos_idx = vocab.stoi['bos']
# bidir = False
# tie_weights = True

# model = LSTM_LM(d_vocab, d_embedding, d_hidden, n_layers,
#                 lstm_drop, lin_drop, bos_idx, bidir, tie_weights)

# learn = Learner(dl, model, loss_func=CrossEntropyLossFlat())

# learn = learn.to_fp16()

# print('training')

# learn.fit_one_cycle(1, 1e-3)

# # torch.save(learn.model.state_dict(), '../nbs/untracked_files/lstm_lm_zinc.pt')
# torch.save(learn.model.state_dict(), '../nbs/untracked_files/lstm_large_lm_zinc.pt')

