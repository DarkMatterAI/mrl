import sys
sys.path.append('../')

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

os.environ['DEFAULT_GPU'] = '1'
os.environ['ncpus'] = '16'



df = pd.read_csv('../../smiles_datasets/chembl/chembl.csv', usecols=['smiles', 'smiles_nc'])

df = df[df.smiles.map(lambda x: len(x)<=90)]

vocab = CharacterVocab(SMILES_CHAR_VOCAB)

def vector_reconstruction_collate2(batch, pad_idx, batch_first=True):

    x,y = vector_reconstruction_collate(batch, pad_idx, batch_first)
    x = x[::-1]
    output = (x,y)
    return output

collate = partial(vector_reconstruction_collate2, pad_idx=vocab.stoi['pad'])
# ds = Vec_Recon_Dataset(['CCC'], vocab, ECFP6, collate_function=collate)
ds = Vec_Recon_Dataset(df.smiles.values, vocab, partial(failsafe_fp, fp_function=ECFP6), collate_function=collate)

d_vocab = len(vocab.itos)
d_embedding = 256
encoder_d_in = 2048
encoder_dims = [1024, 512]
encoder_drops = [0.1, 0.1]
d_hidden = 1024
n_layers = 3
d_latent = 512
dec_drop=0.0
condition_hidden=True
condition_output=True
prior=None
bos_idx=0

lm_model = MLP_VAE(d_vocab, d_embedding, encoder_d_in, encoder_dims, encoder_drops,
                d_hidden, n_layers, d_latent, dec_drop, condition_hidden, condition_output,
                prior, bos_idx)

class VAELoss():
    def __init__(self, weight=1.):
        self.ce = CrossEntropy()
        self.weight = weight
        
    def __call__(self, inputs, targs):
        output, kl_loss = inputs
        return self.ce(output, targs) + self.weight*kl_loss

loss = VAELoss(weight=0.1)

agent = GenerativeAgent(lm_model, vocab, loss, ds, base_model=None)

agent.train_supervised(400, 1, 1e-3)

