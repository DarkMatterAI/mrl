# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_torch_core.ipynb (unless otherwise specified).

__all__ = ['get_device', 'to_device', 'set_device', 'get_model_device', 'USE_CUDA', 'x_to_preds', 'get_log_probs',
           'average_batches', 'smooth_batches', 'pad_and_merge', 'merge_weights', 'merge_models', 'CrossEntropy',
           'HuberLoss']

# Cell
from .imports import *
from .torch_imports import *

# Cell

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    torch.backends.cudnn.benchmark = True

def get_device():
    if torch.cuda.is_available():
        device = int(os.environ.get('DEFAULT_GPU') or 0)
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
    os.environ['DEFAULT_GPU'] = device

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

def get_log_probs(x, y):
    lps = F.log_softmax(x, -1)
    lps = lps.gather(2, y.unsqueeze(-1)).squeeze(-1)
    return lps

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

class CrossEntropy():
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        output = output.view(-1, output.shape[-1])
        target = target.view(-1)
        return self.loss(output, target)

class HuberLoss():
    def __init__(self, beta=1.):
        self.loss = nn.SmoothL1Loss(beta=beta)

    def __call__(self, output, target):
        return self.loss(output, target)