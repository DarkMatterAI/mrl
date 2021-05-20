# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/08_agent.ipynb (unless otherwise specified).

__all__ = ['Agent', 'PredictiveAgent', 'GenerativeAgent', 'ModelOutput']

# Cell
from .imports import *
from .core import *
from .torch_imports import *
from .torch_core import *

# Cell

class Agent():
    def __init__(self, model, loss_function, dataset, opt_kwargs={}):
        self.model = model

        to_device(self.model)

        self.dataset = dataset
        self.opt = self.get_opt(self.model, **opt_kwargs)
        self.loss_function = loss_function

    def get_opt(self, model, **optim_kwargs):
        return optim.Adam(model.parameters(), **optim_kwargs)

    def one_batch(self, batch):
        x,y = batch
        if not type(x)==list:
            x = [x]
        output = self.model(*x)
        loss = self.loss_function(output, y)
        return loss

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.opt.step()

    def train_supervised(self, bs, epochs, lr, percent_valid=0.05):

        train_ds, valid_ds = self.dataset.split(percent_valid)

        train_dl = train_ds.dataloader(bs, shuffle=True)
        valid_dl = valid_ds.dataloader(bs)

        scheduler = optim.lr_scheduler.OneCycleLR(self.opt, max_lr=lr,
                                                 steps_per_epoch=len(train_dl), epochs=10)

        mb = master_bar(range(epochs))
        mb.write(['Epoch', 'Train Loss', 'Valid  Loss', 'Time'], table=True)
        for epoch in mb:
            start = time.time()
            train_losses = []

            for batch in progress_bar(train_dl, parent=mb):

                loss = self.one_batch(batch)

                self.zero_grad()
                loss.backward()
                self.step()
                scheduler.step()
                train_losses.append(loss.detach().cpu())
                mb.child.comment = f"{train_losses[-1]:.5f}"

            with torch.no_grad():
                valid_losses = []
                for batch in progress_bar(valid_dl, parent=mb):

                    loss = self.one_batch(batch)
                    valid_losses.append(loss.detach().cpu())
                    mb.child.comment = f"{valid_losses[-1]:.5f}"

            train_loss = smooth_batches(train_losses)
            valid_loss = smooth_batches(valid_losses)
            end = time.time() - start
            mb.write([epoch, f'{train_losses[-1]:.5f}',
                      f'{valid_losses[-1]:.5f}', f'{format_time(end)}'], table=True)

    def update_dataset(self, dataset):
        self.dataset = dataset

    def update_dataset_from_inputs(self, *dataset_inputs):
        dataset = self.dataset.new(*dataset_inputs)
        self.update_dataset(dataset)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=get_model_device(self.model))

        self.model.load_state_dict(state_dict)

    def save_weights(self, filename):

        state_dict = self.base_model.state_dict()
        torch.save(state_dict, filename)

# Cell

class PredictiveAgent(Agent):

    def predict_tensor(self, x):
        if not type(x)==list:
            x = [x]
        output = self.model(*x)

    def predict_data(self, data):
        ds = self.dataset.new(data, [0 for i in data])
        batch = ds.collate_function([ds[i] for i in range(len(ds))])
        x,y = batch
        return self.predict_tensor(x)

# Cell

class GenerativeAgent(Agent):
    def __init__(self, model, vocab, loss_function, dataset,
                 base_model=True, value_head=None, latents=None,
                 opt_kwargs={}, vopt_kwargs={}, lopt_kwargs={}):
        super().__init__(model, loss_function, dataset, opt_kwargs)

        if base_model==True:
            self.base_model = copy.deepcopy(model)
        else:
            self.base_model = base_model

        to_device(self.model)
        to_device(self.base_model)

        self.vocab = vocab
        self.value_head = value_head
        self.latents = latents

        self.opts = [self.opt]
        if self.value_head is not None:
            to_device(self.value_head)
            self.value_opt = self.get_opt(self.value_head, **vopt_kwargs)
            self.opts.append(self.value_opt)

        if self.latents is not None:
            to_device(self.latents)
            self.latent_opt = self.get_opt(self.latents, **lopt_kwargs)
            self.opts.append(self.latent_opt)

    def zero_grad(self):
        for opt in self.opts:
            opt.zero_grad()

    def step(self):
        for opt in self.opts:
            opt.step()

    def reconstruct(self, preds):
        return maybe_parallel(self.vocab.reconstruct, [i for i in preds.detach().cpu()])

    def load_weights(self, filename, base=False):
        state_dict = torch.load(filename, map_location=get_model_device(self.model))

        if 'value_head' in state_dict.keys():
            value_state = state_dict['value_head']
            state_dict = state_dict['model']
            if not base:
                self.value_head.load_state_dict(value_state)

        if not base:
            self.model.load_state_dict(state_dict)
        else:
            if not isinstance(self.base_model, nn.Module):
                self.base_model = copy.deepcopy(model)

            self.base_model.load_state_dict(state_dict)

    def save_weights(self, filename, base=False):

        if base:
            state_dict = self.base_model.state_dict()
        else:
            state_dict = self.model.state_dict()

        if self.value_head is not None:
            value_state = self.value_head.state_dict()
        else:
            value_state = None

        state_dict = {'model':state_dict, 'value_head':value_state}

        torch.save(state_dict, filename)

    def get_batch_params(self, model_output):
        x = model_output['x']
        y = model_output['y']
        mask = ~(y==self.vocab.stoi['pad'])
        lengths = mask.sum(-1)
        sl = y.shape[-1]
        smiles = self.reconstruct(y)

        model_output['mask'] = mask
        model_output['lengths'] = lengths
        model_output['sl'] = sl
        model_output['sequences'] = smiles

        return model_output


    def get_model_outputs(self, model_output):
        x = model_output['x']
        y = model_output['y']
        latent = model_output['latent']
        mo, mlp, mglp, me = self.model.get_rl_tensors(x,y,latent=latent)
        mprob = mlp.exp()

        model_output['model_output'] = mo
        model_output['model_logprobs'] = mlp
        model_output['model_gathered_logprobs'] = mglp
        model_output['model_encoded'] = me
        model_output['y_gumbel'] = F.one_hot(y, len(self.vocab.itos)) + mprob - mprob.detach()

        if self.value_head is not None:
            value_predictions = self.value_head(me)
        else:
            value_predictions = None

        model_output['state_values'] = value_predictions

        if self.base_model is not None:
            with torch.no_grad():
                bo, blp, bglp, be = self.base_model.get_rl_tensors(x,y)
        else:
            bo, blp, bglp, be = None, None, None, None

        model_output['reference_output'] = bo
        model_output['reference_logprobs'] = blp
        model_output['reference_gathered_logprobs'] = bglp
        model_output['reference_encoded'] = be

        return model_output

# Cell

class ModelOutput(dict):
    def __init__(self):
        super().__init__()

        self.__setitem__('sequences', [])                      # buffer/batch
        self.__setitem__('mols', [])                           # buffer/batch
        self.__setitem__('source', [])                         # buffer/batch
        self.__setitem__('x', None)                            # buffer/batch
        self.__setitem__('y', None)                            # buffer/batch
        self.__setitem__('mask', None)                         # buffer/batch
        self.__setitem__('lengths', None)                      # buffer/batch
        self.__setitem__('sl', None)                           # buffer/batch
        self.__setitem__('model_output', None)                 # model
        self.__setitem__('model_encoded', None)                # model
        self.__setitem__('model_logprobs', None)               # model
        self.__setitem__('model_gathered_logprobs', None)      # model
        self.__setitem__('y_gumbel', None)                     # agent
        self.__setitem__('latent', None)                       # agent
        self.__setitem__('state_values', None)                 # agent value_head
        self.__setitem__('reference_output', None)             # reference model
        self.__setitem__('reference_encoded', None)            # reference model
        self.__setitem__('reference_logprobs', None)           # reference model
        self.__setitem__('reference_gathered_logprobs', None)  # reference model
        self.__setitem__('rewards', None)                      # reward function
        self.__setitem__('rewards_dict', {})                   # reward function
        self.__setitem__('rewards_scaled', None)               # reward function
        self.__setitem__('trajectory_rewards', None)           # trajectory reward function
        self.__setitem__('losses',     {'pg_loss' : None,
                                       'diff_loss' : None})
        self.__setitem__('loss_dicts', {'pg_dict' : {},
                                        'diff_dict' : {}})


