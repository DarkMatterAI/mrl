# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_torch_core.ipynb (unless otherwise specified).

__all__ = ['get_device', 'to_device', 'set_device', 'get_model_device', 'USE_CUDA', 'freeze', 'unfreeze', 'x_to_preds',
           'gather_lps', 'merge_weights', 'merge_models', 'CrossEntropy']

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

def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad_(True)

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

# Cell

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
        target = target.view(-1).long()
        return self.loss(output, target)