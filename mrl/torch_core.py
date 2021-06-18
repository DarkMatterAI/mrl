# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_torch_core.ipynb (unless otherwise specified).

__all__ = ['get_device', 'to_device', 'set_device', 'get_model_device', 'USE_CUDA', 'x_to_preds', 'gather_lps',
           'gumbel_onehot', 'average_batches', 'smooth_batches', 'pad_and_merge', 'merge_weights', 'merge_models',
           'freeze', 'unfreeze', 'discount_rewards', 'whiten', 'scatter_rewards', 'compute_advantages', 'CrossEntropy',
           'BinaryCrossEntropy', 'HuberLoss', 'MSELoss', 'pca']

# Cell
from .imports import *
from .torch_imports import *

# Cell

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    torch.backends.cudnn.benchmark = True


if USE_CUDA:
    os.environ['use_cuda'] = 'cuda'
else:
    os.environ['use_cuda'] = 'cpu'

def get_device():
    if torch.cuda.is_available() and os.environ['use_cuda']=='cuda':
        device = torch.cuda.current_device()
    else:
        device='cpu'

    device = torch.device(device)
    return device

def to_device(tensor, device=None):
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    if hasattr(tensor, 'to'):
        output = tensor.to(device)
    else:
        output = [to_device(i, device) for i in tensor]

    return output

def set_device(device):
    torch.cuda.set_device(device)

def get_model_device(model):
    return next(model.parameters()).device

# Cell

def x_to_preds(x, multinomial=True):
    log_probs = F.log_softmax(x, -1).squeeze(1)
    probs = log_probs.detach().exp()

    if multinomial:
        idxs = torch.multinomial(probs, 1)
    else:
        idxs = x.argmax(-1)

    lps = torch.gather(log_probs, 1, idxs)
    return idxs, lps

def gather_lps(lps, y):
    return lps.gather(2, y.unsqueeze(-1)).squeeze(-1)

def gumbel_onehot(y, num_classes=-1, probs=None, log_probs=None):

    y_onehot = F.one_hot(y, num_classes)

    if probs is not None:
        y_onehot = y_onehot + probs - probs.detach()

    if log_probs is not None:
        y_onehot = y_onehot + log_probs - log_probs.detach()

    return y_onehot

# Cell

def average_batches(batches):
    val = torch.tensor(0.)
    c = 0
    for batch in batches:
        bs = batch.shape[0]
        val += batch.mean()*bs
        c += bs

    return val/c

def smooth_batches(batches, beta=0.98):

    val = torch.tensor(0.)
    count = 0

    for batch in batches:
        val = torch.lerp(batch.mean(), val, beta)
        count += 1

    return val/(1-beta**count)


# Cell

def pad_and_merge(x1, x2, pad_idx, batch_dim=0, pad_dim=1):

    if type(x1)==list:
        output = [pad_and_merge(x1[i], x2[i], pad_idx, batch_dim, pad_dim)
                  for i in range(len(x1))]
    else:
        new_bs = x1.shape[batch_dim] + x2.shape[batch_dim]
        new_sl = max(x1.shape[pad_dim], x2.shape[pad_dim])

        output = torch.zeros((new_bs, new_sl)).long() + pad_idx
        output = output.type(x1.dtype)
        output = output.to(x1.device)
        output[:x1.shape[0], :x1.shape[1]] = x1
        output[x1.shape[0]:, :x2.shape[1]] = x2

    return output

def merge_weights(sd1, sd2, alpha=0.5):
    new_dict = {}
    for key in sd1.keys():
        if key in sd2.keys():
            new_dict[key] = sd1[key]*alpha + sd2[key]*(1-alpha)
        else:
            new_dict[key] = sd1[key]
    return new_dict

def merge_models(model1, model2, alpha=0.5):
    new_weights = merge_weights(model1.state_dict(), model2.state_dict(), alpha)
    model1.load_state_dict(new_weights)

# Cell

def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad_(True)

# Cell

def discount_rewards(rewards, gamma):
    discounted = torch.zeros((rewards.shape[0], rewards.shape[1]+1)).to(rewards.device)

    for i in reversed(range(discounted.shape[1]-1)):
        discounted[:,i] = rewards[:,i] + gamma*discounted[:,i+1]

    return discounted[:,:-1]

# def whiten(values, shift_mean=True):
#     mean, var = torch.mean(values), torch.var(values)
#     whitened = (values - mean) * torch.rsqrt(var + 1e-8)
#     if not shift_mean:
#         whitened += mean
#     return whitened

def whiten(values, shift_mean=True, mask=None):
    if mask is None:
        mean = values.mean()
        var = values.var()
    else:
        mean = (values*mask).sum()/mask.sum()
        var = ((values-mean)*mask).pow(2).sum()/(mask.sum()-1)

    whitened = (values - mean) * torch.rsqrt(var + 1e-8)

    if not shift_mean:
        whitened += mean

    if mask is not None:
        whitened = whitened*mask

    return whitened

def scatter_rewards(rewards, mask):
    template = torch.zeros(mask.shape).to(mask.device)
    lengths = mask.sum(-1)
    template[torch.arange(template.shape[0]), lengths-1]=rewards
    return template

def compute_advantages(rewards, values, gamma, lam):

    advantages = torch.zeros(rewards.shape).to(rewards.device)

    for i in reversed(range(rewards.shape[1])):
        if i==rewards.shape[1]-1:
            v_t1 = 0.
            glv = 0.
        else:
            v_t1 = values[:,i+1]

        delta = rewards[:, i] + gamma*v_t1 - values[:, i]
        glv = delta + gamma * lam * glv
        advantages[:,i] = glv

    return advantages

# Cell

class CrossEntropy():
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        output = output.view(-1, output.shape[-1])
        target = target.view(-1).long()
        return self.loss(output, target)

class BinaryCrossEntropy():
    def __init__(self):
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, output, target):
        output = output.view(-1)
        target = target.view(-1)
        return self.loss(output, target)

class HuberLoss():
    def __init__(self, beta=1.):
        self.loss = nn.SmoothL1Loss(beta=beta)

    def __call__(self, output, target):
        output = output.squeeze(-1)
        return self.loss(output, target)


class MSELoss():
    def __init__(self):
        self.loss = F.mse_loss

    def __call__(self, output, target):
        output = output.squeeze(-1)
        return self.loss(output, target)

# Cell

def pca(x, k=2):
    x = x-torch.mean(x,0)
    U,S,V = torch.svd(x.t())
    return torch.mm(x,U[:,:k])