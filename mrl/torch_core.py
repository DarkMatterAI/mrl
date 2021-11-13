# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_torch_core.ipynb (unless otherwise specified).

__all__ = ['get_device', 'to_device', 'set_device', 'get_model_device', 'USE_CUDA', 'freeze', 'unfreeze', 'x_to_preds',
           'gather_lps', 'subset_tensor', 'merge_weights', 'merge_models', 'smooth_batches', 'discount_rewards',
           'whiten', 'scatter_rewards', 'compute_advantages', 'CrossEntropy', 'BinaryCrossEntropy', 'HuberLoss',
           'MSELoss', 'pca']

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
    '''
    get_device - returns current default device.

    If Cuda is available and `os.environ['use_cuda']=='cuda'`,
    `torch.cuda.current_device()` is returned.

    Otherwise, `cpu` is returned

    Returns `torch.device`
    '''
    if torch.cuda.is_available() and os.environ['use_cuda']=='cuda':
        device = torch.cuda.current_device()
    else:
        device='cpu'

    device = torch.device(device)
    return device

def to_device(tensor, device=None):
    '''
    to_device - sets `tensor` to `device` if possible.
    If `device=None`, `tensor` is set to the default device
    returned by `get_device`

    Inputs

    - `tensor torch.Tensor`: input tensor

    - `device [str, torch.Device]`: device
    '''
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
    'wrapper for torch set_device'
    torch.cuda.set_device(device)

def get_model_device(model):
    'gets device for model from first parameter'
    return next(model.parameters()).device

# Cell

def freeze(module):
    '''
    freeze - freezes all parameters in `module` (requires_grad=False)

    Inputs:

    - `module nn.Module`: Pytorch module
    '''
    for p in module.parameters():
        p.requires_grad_(False)

def unfreeze(module):
    '''
    unfreeze - unfreezes all parameters in `module` (requires_grad=True)

    Inputs:

    - `module nn.Module`: Pytorch module
    '''
    for p in module.parameters():
        p.requires_grad_(True)

# Cell

def x_to_preds(x, multinomial=True):
    '''
    x_to_preds - helper function for converting `x`
    to log probs and taking a hard sample

    Inputs:

    - `x torch.Tensor`: input tensor

    - `multinomial bool`: if True, use multinomial sampling.
    If False, use argmax sampling

    Returns:

    - `idxs torch.LongTensor`: index values of hard sample

    - `lps torch.FloatTensor`: log probabilities for each hard value
    '''
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

# Cell

def subset_tensor(x, mask):
    '''
    indexes `x` with `mask`. If `x` is a list or tuple,
    function will index all items in `x`
    '''

    if isinstance(x, (list, tuple)):
        x = [i[mask] for i in x]
    else:
        x = x[mask]

    return x

# Cell

def merge_weights(sd1, sd2, alpha=0.5):
    '''
    merges state dicts following `new_weight = alpha*weight_model1 + (1-alpha)*weight_model2`

    New weights are returned as a new state dict
    '''
    new_dict = {}
    for key in sd1.keys():
        if key in sd2.keys():
            new_dict[key] = sd1[key]*alpha + sd2[key]*(1-alpha)
        else:
            new_dict[key] = sd1[key]
    return new_dict

def merge_models(model1, model2, alpha=0.5):
    '''
    merges weights following `new_weight = alpha*weight_model1 + (1-alpha)*weight_model2`

    New weights are loaded into `modell` inplace
    '''
    new_weights = merge_weights(model1.state_dict(), model2.state_dict(), alpha)
    model1.load_state_dict(new_weights)

# Cell

def smooth_batches(batches, beta=0.98):
    val = torch.tensor(0.)
    count = 0

    for batch in batches:
        val = torch.lerp(batch.mean(), val, beta)
        count += 1

    return val/(1-beta**count)

# Cell

def discount_rewards(rewards, gamma):
    '''
    discount_rewards - discounts `rewards` by `gamma`

    Inputs:

    - `rewards torch.Tensor[bs,sl]`: tensor of undiscounted rewards

    - `gamma float`: discount factor

    Returns:

    - `discounted torch.Tensor[bs, sl]`: tensor of discounted rewards

    Rewards are discounted following

    `discounted[i] = rewards[i] + gamma*discounted[i+1]`
    '''
    discounted = torch.zeros((rewards.shape[0], rewards.shape[1]+1)).to(rewards.device)

    for i in reversed(range(discounted.shape[1]-1)):
        discounted[:,i] = rewards[:,i] + gamma*discounted[:,i+1]

    return discounted[:,:-1]

# Cell

def whiten(values, shift_mean=True, mask=None):
    '''
    whiten - whitens `values`

    Inputs:

    - `values torch.FloatTensor`: values to be whitened

    - `shift_mean bool`: if True, outputs will have zero mean.

    - `mask [torch.BoolTensor, torch.LongTensor, None]`: if a mask
    is given, masked values will not contribute to calculating the
    mean and variance for whitening. Masking is done following
    `masked_values = values*mask`. For bool tensors, values where
    `mask=True` are kept. For binary float/int tensors, values
    where `mask=1` are kept.

    Returns:

    - `whitened torch.FloatTensor`: whitened values
    '''
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

# Cell

def scatter_rewards(rewards, mask):
    '''
    scatter_rewards - scatter vector of rewards to matrix
    based on `mask`

    Inputs:

    - `rewards torch.FloatTensor[bs]`: vector of rewards

    - `mask torch.Tensor[bs, sl]`: mask tensor

    Returns:

    - `template torch.FloatTensor`: scattered values

    In molecular RL, we typically have a single reward per molecule
    evaluating the entire structure. Before our update, we need to
    discount the final reward back to all timesteps. However, we want
    to ignore padding when we do this.

    scatter_rewards takes a vector of rewards and a mask where
    non-padding tokens are True and padding tokens are False.
    Rewards are placed in the last `True` index

    ```
    rewards = torch.tensor([4., 5., 6.]).float()
    mask = torch.tensor([[True, True, True, False],
                         [True, True, False, False],
                         [True, True, True, True]])
    scattered = scatter_rewards(rewards, mask)
    >> torch.tensor([[0., 0., 4., 0.],
                     [0., 5., 0., 0.],
                     [0., 0., 0., 6.]])
    ```
    '''
    template = torch.zeros(mask.shape).to(mask.device)
    lengths = mask.sum(-1)
    template[torch.arange(template.shape[0]), lengths-1]=rewards
    return template


# Cell

def compute_advantages(rewards, values, gamma, lam):
    '''
    Calculate advantages according to Generalized Advantage Estimation (GAE)

    Inputs:

    - `rewards torch.Tensor`: reward tensor

    - `values torch.Tensor`: value function predictions

    - 'gamma float`: GAE gamma factor

    - `lam float`: GAE lambda factor

    Returns:

    - `advantages torch.Tensor`: computed advantages

    Advantages are computed according to GAE

    `delta = rewards[i] + gamma*values[i+1] - values[i]`

    `advantages[i] = delta + gamma*lam*glv`
    '''

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
    '''
    CrossEntropy - cross entropy loss for sequence predictions.
    Flattens predictions and targets before computing loss
    '''
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        '''
        Inputs:

        - `output torch.FloatTensor[bs, sl, n]`: predictions

        - `target torch.LongTensor[bs, sl]`: target integer values
        '''
        if USE_CUDA:
            output = output.view(-1, output.shape[-1])
            target = target.view(-1).long()
        else:
            output = output.reshape(-1, output.shape[-1])
            target = target.reshape(-1).long()
        return self.loss(output, target)

class BinaryCrossEntropy():
    def __init__(self):
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, output, target):
        if USE_CUDA:
            output = output.view(-1)
            target = target.view(-1)
        else:
            output = output.reshape(-1)
            target = target.reshape(-1)
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
    '''
    pca of `x` to `k` dimensions
    '''
    x = x-torch.mean(x,0)
    U,S,V = torch.svd(x.t())
    return torch.mm(x,U[:,:k])