# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/16_agent.ipynb (unless otherwise specified).

__all__ = ['Agent', 'PredictiveAgent', 'BaselineAgent', 'CriticAgent', 'GenerativeAgent', 'SupervisedCB', 'Rollback',
           'RetrainRollback', 'ResetAndRetrain', 'MetricResetAndRetrain', 'SaveAgentWeights']

# Cell

from ..imports import *
from ..core import *
from ..torch_imports import *
from ..torch_core import *
from .callback import *

# Cell

class Agent(Callback):
    '''
    Agent - class for bundling a model, loss function, and dataset

    Inputs:

    - `model nn.Module`: model

    - `loss_function Callable`: loss function for supervised training. Should
    function as `loss = loss_function(model_output, y)`

    - `dataset Base_Dataset`: dataset

    - `opt_kwargs dict`: dictionary of keyword arguments passed to `optim.Adam`

    - `clip float`: gradient clipping

    - `name str`: agent name
    '''
    def __init__(self, model, loss_function, dataset, opt_kwargs={}, clip=1., name='agent'):
        super().__init__(name=name, order=2)

        self.model = model
        to_device(self.model)

        self.loss_function = loss_function
        self.dataset = dataset

        self.opt = self.get_opt(self.model.parameters(), **opt_kwargs)
        self.clip = clip
        self.training = True
        self.compute_outputs = True
        self.best_weights = None
        self.best_loss = float('inf')
        self.best_metric = float('-inf')

    def get_opt(self, parameters, **optim_kwargs):
        return optim.Adam(parameters, **optim_kwargs)

    def before_compute_reward(self):
        '''
        uses self.dataset to convert samples into tensors
        '''
        env = self.environment
        batch_state = env.batch_state
        sequences = batch_state.samples

        batch_ds = self.dataset.new(sequences)
        batch = batch_ds.collate_function([batch_ds[i] for i in range(len(batch_ds))])
        batch = to_device(batch)
        bs = len(batch_ds)
        x,y = batch

        batch_state.x = x
        batch_state.y = y
        batch_state.bs = bs
#         batch_state.rewards = to_device(torch.zeros(bs))

    def zero_grad(self):
        self.opt.zero_grad()

    def before_step(self):
        if self.training:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

    def step(self):
        if self.training:
            self.opt.step()

    def one_batch(self, batch, fp16=False, return_output=False):
        batch = to_device(batch)
        x,y = batch
        if not isinstance(x, (list, tuple)):
            x = [x]

        if fp16:
            with torch.cuda.amp.autocast():
                output = self.model(*x)
                loss = self.loss_function(output, y)
        else:
            output = self.model(*x)
            loss = self.loss_function(output, y)

        if return_output:
            return loss, output
        else:
            return loss

    def load_best(self):
        if self.best_weights is not None:
            self.load_state_dict(self.best_weights)

    def train_supervised(self, bs, epochs, lr, valid_metric=None,
                         percent_valid=0.05, silent=False,
                         fp16=False, save_every=None,
                         load_best_every=None, opt_kwargs={}):
        '''
        train_supervised - trains on data in `self.dataset`

        Inputs:

        - `bs int`: batch size

        - `epochs int`: number of epochs

        - `lr float`: learning rate passed to `optim.lr_scheduler.OneCycleLR`

        - `valid_metric Optional[Callable]`: a callable validation metric
        in the form of `valid_metric(predictions, targets)`. If given, will
        be calculated for the validation set every epoch

        - `percent_valid float`: validation set percentage

        - `silent bool`: if training losses should be printed

        - `fp16 bool`: if FP16 training should be used

        - `save_every Optional[int]`: If an integer is given, model
        weights are saved every `save_every` batches to
        `{self.name}_weights_{batch}.pt`

        - `load_best_every Optional[int]`: if integer, the best model
        weights by validation loss are loaded every `load_best_every` batches

        - `opt_kwargs Optional[dict]`: keyword arguments passed to optimzier
        '''

        total_batches = 0

        if fp16:
            scaler = torch.cuda.amp.GradScaler()

        train_ds, valid_ds = self.dataset.split(percent_valid)

        if len(train_ds)%bs==1:
            train_dl = train_ds.dataloader(bs, shuffle=True, drop_last=True)
        else:
            train_dl = train_ds.dataloader(bs, shuffle=True)

        valid_dl = valid_ds.dataloader(bs)

        opt = optim.Adam(self.model.parameters(), lr=lr, **opt_kwargs)

        scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                        steps_per_epoch=len(train_dl), epochs=epochs)

        if silent:
            mb = range(epochs)
        else:
            mb = master_bar(range(epochs))
            cols = ['Epoch', 'Train Loss', 'Valid  Loss', 'Time']
            if valid_metric is not None:
                cols.append(valid_metric.__name__)
            mb.write(cols, table=True)

        for epoch in mb:
            start = time.time()
            train_losses = []

            if silent:
                batch_iter = iter(train_dl)
            else:
                batch_iter = progress_bar(train_dl, parent=mb)

            for batch in batch_iter:

                loss = self.one_batch(batch, fp16=fp16)
                opt.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                scheduler.step()
                train_losses.append(loss.detach().cpu())

                if not silent:
                    mb.child.comment = f"{train_losses[-1]:.5f}"

                total_batches += 1

                if (load_best_every is not None) and (total_batches%load_best_every==0):
                    self.load_best()

                if (save_every is not None) and (total_batches%save_every==0):
                    self.save_weights(f'{self.name}_weights_{total_batches}.pt')

            with torch.no_grad():
                self.model.eval()
                valid_losses = []
                preds = []
                targs = []

                if len(valid_ds)>0:
                    if silent:
                        batch_iter = iter(valid_dl)
                    else:
                        batch_iter = progress_bar(valid_dl, parent=mb)

                    for batch in batch_iter:

                        loss, p = self.one_batch(batch, return_output=True)
                        valid_losses.append(loss.detach().cpu())
                        x,y = batch
                        preds.append(p.detach().cpu())
                        targs.append(y.detach().cpu())

                        if not silent:
                            mb.child.comment = f"{valid_losses[-1]:.5f}"
                    preds = torch.cat(preds)
                    targs = torch.cat(targs)
                else:
                    valid_losses = [torch.tensor(0.)]
                self.model.train()

            train_loss = smooth_batches(train_losses)
            valid_loss = smooth_batches(valid_losses)
            end = time.time() - start

            if valid_metric is not None:
                metric = valid_metric(preds, targs)
                if metric < self.best_metric:
                    self.best_metric = metric
                    self.best_weights = copy.deepcopy(self.model).cpu().state_dict()
            else:
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.best_weights = copy.deepcopy(self.model).cpu().state_dict()

            if not silent:
                epoch_data = [epoch, f'{train_loss:.5f}', f'{valid_loss:.5f}', f'{format_time(end)}']
                if valid_metric is not None:
                    epoch_data.append(f'{metric}:.5f')
                mb.write(epoch_data, table=True)

    def update_dataset(self, dataset):
        self.dataset = dataset

    def update_dataset_from_inputs(self, *dataset_inputs):
        dataset = self.dataset.new(*dataset_inputs)
        self.update_dataset(dataset)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=get_model_device(self.model))
        self.load_state_dict(state_dict)

    def save_weights(self, filename):
        state_dict = self.model.state_dict()
        torch.save(state_dict, filename)

    def save(self, filename):
        torch.save(self, filename)


# Cell

class PredictiveAgent(Agent):
    '''
    PredictiveAgent - Agent class for predictive models

    Inputs:

    - `model nn.Module`: model

    - `loss_function Callable`: loss function for supervised training. Should
    function as `loss = loss_function(model_output, y)`

    - `dataset Base_Dataset`: dataset

    - `opt_kwargs dict`: dictionary of keyword arguments passed to `optim.Adam`

    - `clip float`: gradient clipping

    - `name str`: agent name
    '''

    def predict_tensor(self, x):
        with torch.no_grad():
            if not isinstance(x, (list, tuple)):
                x = [x]
            output = self.model(*x)
            return output

    def predict_dataset(self, dataset):
        batch = dataset.collate_function([dataset[i] for i in range(len(dataset))])
        batch = to_device(batch)
        x,y = batch
        return self.predict_tensor(x)

    def predict_dataset_batch(self, dataset, bs, **dl_kwargs):
        dl = dataset.dataloader(bs, shuffle=False, **dl_kwargs)
        preds = []
        for i, batch in enumerate(dl):
            x,y = batch
            x = to_device(x)
            p = self.predict_tensor(x)
            preds.append(p.detach())
        preds = torch.cat(preds)
        return preds

    def predict_data(self, data):
        ds = self.dataset.new(data, [0 for i in data])
        return self.predict_dataset(ds)

    def predict_data_batch(self, data, bs, **dl_kwargs):
        ds = self.dataset.new(data, [0 for i in data])
        return self.predict_dataset_batch(ds, bs, **dl_kwargs)


# Cell

class BaselineAgent(Agent):
    '''
    BaselineAgent - agent for a model with a baseline model

    Inputs:

    - `model nn.Module`: model

    - `loss_function Callable`: loss function for supervised training. Should
    function as `loss = loss_function(model_output, y)`

    - `dataset Base_Dataset`: dataset

    - `base_update float`: update fraction for the baseline model. Updates
    the base model following `base_model = base_update*base_model + (1-base_update)*model`

    - `base_update_iter int`: update frequency for baseline model

    - `base_model bool`: if False, baseline model will not be created

    - `opt_kwargs dict`: dictionary of keyword arguments passed to `optim.Adam`

    - `clip float`: gradient clipping

    - `name str`: agent name
    '''
    def __init__(self, model, loss_function, dataset, base_update=0.99,
                 base_update_iter=10, base_model=True, opt_kwargs={},
                 clip=1., name='baseline_agent'):
        super().__init__(model, loss_function, dataset, opt_kwargs, clip, name)

        self.set_models(base_model)
        self.base_update = base_update
        self.base_update_iter = base_update_iter

    def after_batch(self):
        log = self.environment.log
        iterations = log.iterations
        if iterations%self.base_update_iter == 0 and iterations>0:
            self.update_base_model()

    def set_models(self, base_model):

        if base_model==True:
            self.base_model = copy.deepcopy(self.model)
        else:
            self.base_model = base_model

        try:
            to_device(self.base_model)
        except:
            pass

    def base_to_model(self):
        '''
        copies weights from `model` into `base_model`
        '''
        if type(self.base_model)==type(self.model):
            self.base_model.load_state_dict(self.model.state_dict())

    def model_to_base(self):
        '''
        copies weights from `base_model` into `model`
        '''
        if type(self.base_model)==type(self.model):
            self.model.load_state_dict(self.base_model.state_dict())

    def update_base_model(self):
        '''
        updates baseline model weights
        '''
        if type(self.base_model)==type(self.model):
            if self.base_update < 1:
                merge_models(self.base_model, self.model, alpha=self.base_update)

    def save_weights(self, filename):
        state_dict = {}
        state_dict['model'] = self.model.state_dict()

        if isinstance(self.base_model, nn.Module):
            state_dict['base_model'] = self.base_model.state_dict()
        else:
            state_dict['base_model'] = None

        torch.save(state_dict, filename)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])

        if isinstance(self.base_model, nn.Module):
            self.base_model.load_state_dict(state_dict['base_model'])

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=get_model_device(self.model))
        self.load_state_dict(state_dict)


# Cell

class CriticAgent(BaselineAgent):
    '''
    CriticAgent - baseline agent for critic models

    Inputs:

    - `model nn.Module`: model

    - `loss_function Callable`: loss function for supervised training. Should
    function as `loss = loss_function(model_output, y)`

    - `dataset Base_Dataset`: dataset

    - `base_update float`: update fraction for the baseline model. Updates
    the base model following `base_model = base_update*base_model + (1-base_update)*model`

    - `base_update_iter int`: update frequency for baseline model

    - `base_model bool`: if False, baseline model will not be created

    - `opt_kwargs dict`: dictionary of keyword arguments passed to `optim.Adam`

    - `clip float`: gradient clipping

    - `name str`: agent name
    '''
    def predict_tensor(self, x, baseline=False):
        if not type(x)==list:
            x = [x]

        if baseline:
            if isinstance(self.base_model, nn.Module):
                output = self.base_model(*x)
            else:
                output = None
        else:
            output = self.model(*x)

        return output

    def predict_data(self, data):
        ds = self.dataset.new(data, [0 for i in data])
        batch = ds.collate_function([ds[i] for i in range(len(ds))])
        batch = to_device(batch)
        x,y = batch
        return self.predict_tensor(x)

    def get_model_outputs(self):
        if self.compute_outputs:
            env = self.environment
            batch_state = env.batch_state
            x = batch_state.x
            y = batch_state.y

            preds = self.predict_tensor(x, baseline=False)
            batch_state.model_output = preds

            with torch.no_grad():
                base_preds = self.predict_tensor(x, baseline=True)
                batch_state.base_output = base_preds



# Cell

class GenerativeAgent(BaselineAgent):
    '''
    GenerativeAgent - baseline agent for generative models

    Inputs:

    - `model nn.Module`: model

    - `vocab Vocab`: vocabulary

    - `loss_function Callable`: loss function for supervised training. Should
    function as `loss = loss_function(model_output, y)`

    - `dataset Base_Dataset`: dataset

    - `base_update float`: update fraction for the baseline model. Updates
    the base model following `base_model = base_update*base_model + (1-base_update)*model`

    - `base_update_iter int`: update frequency for baseline model

    - `base_model bool`: if False, baseline model will not be created

    - `opt_kwargs dict`: dictionary of keyword arguments passed to `optim.Adam`

    - `clip float`: gradient clipping

    - `name str`: agent name
    '''
    def __init__(self, model, vocab, loss_function, dataset,
                 base_update=0.99, base_update_iter=10, base_model=True,
                 opt_kwargs={}, clip=1., name='generative_agent'):
        super().__init__(model, loss_function, dataset,
                         base_update=base_update,
                         base_update_iter=base_update_iter,
                         base_model=base_model,
                         opt_kwargs=opt_kwargs,
                         clip=clip,
                         name=name)

        self.vocab = vocab

    def reconstruct(self, preds):
        return maybe_parallel(self.vocab.reconstruct, [i for i in preds.detach().cpu().numpy()])

    def sample_and_reconstruct(self, bs, sl, **sample_kwargs):
        preds, _ = self.model.sample_no_grad(bs, sl, **sample_kwargs)
        recon = self.reconstruct(preds)
        return recon

    def batch_sample_and_reconstruct(self, n_samples, sample_bs, sl,
                                     verbose=False, **sample_kwargs):
        if sample_bs > n_samples:
            sample_bs = n_samples
        n_batches = n_samples//sample_bs
        remainder = n_samples - (n_batches * sample_bs)
        recon = []
        for i in range(n_batches):
            if verbose:
                print(f'batch {i}/{n_batches}')
            recon += self.sample_and_reconstruct(sample_bs, sl, **sample_kwargs)

        if remainder > 0:
            recon += self.sample_and_reconstruct(remainder, sl, **sample_kwargs)
        return recon

    def before_compute_reward(self):
        env = self.environment
        batch_state = env.batch_state
        sequences = batch_state.samples

        batch_ds = self.dataset.new(sequences)
        batch = batch_ds.collate_function([batch_ds[i] for i in range(len(batch_ds))])
        batch = to_device(batch)
        bs = len(batch_ds)
        x,y = batch

        batch_state.x = x
        batch_state.y = y
        batch_state.bs = bs
        mask = ~(y==self.vocab.stoi['pad']) # padding mask
        batch_state.mask = mask
        batch_state.lengths = mask.sum(-1)
        batch_state.sl = y.shape[-1]
#         batch_state.rewards = to_device(torch.zeros(bs))
        batch_state.trajectory_rewards = to_device(torch.zeros(y.shape))

    def get_rl_tensors(self, model, x, y, latent_info, sources):
        '''
        get_rl_tensors - uses latent info to compute output tensors
        '''
        if latent_info:
            latent_sources = []
            output_tensors = []


            for (latent_source, latents) in latent_info.items():
                if latents.shape[0]>0:
                    latent_sources.append(latent_source)
                    latent_mask = torch.tensor([i==latent_source for i in sources]).bool()
                    out = self.model.get_rl_tensors(subset_tensor(x, latent_mask),
                                                          subset_tensor(y, latent_mask),
                                                          latent=latents)
                    out = list(out)
                    out.append(latents)
                    output_tensors.append(out)

            non_latent_mask = torch.tensor([not i in latent_sources for i in sources]).bool()

            if non_latent_mask.sum()>0:
                latents = model.x_to_latent(subset_tensor(x, non_latent_mask))
                out = model.get_rl_tensors(subset_tensor(x, non_latent_mask),
                                           subset_tensor(y, non_latent_mask),
                                           latent=latents)
                out = list(out)
                out.append(latents)
                output_tensors.append(out)

            mo = torch.cat([i[0] for i in output_tensors], 0)
            mlp = torch.cat([i[1] for i in output_tensors], 0)
            mglp = torch.cat([i[2] for i in output_tensors], 0)
            me = torch.cat([i[3] for i in output_tensors], 0)

            if not any([i[4] is None for i in output_tensors]):
                latents = torch.cat([i[4] for i in output_tensors], 0)
            else:
                latents = None

        else:
            latents = model.x_to_latent(x)
            mo, mlp, mglp, me = model.get_rl_tensors(x,y, latent=latents)

        return mo, mlp, mglp, me, latents

    def get_model_outputs(self):

        if self.compute_outputs:
            env = self.environment
            batch_state = env.batch_state

            x = batch_state.x
            y = batch_state.y
            sources = batch_state.sources
            latent_info = batch_state.latent_data

            mo, mlp, mglp, me, ml = self.get_rl_tensors(self.model, x, y, latent_info, sources)
            mprob = mlp.exp()

            batch_state.model_output = mo
            batch_state.model_logprobs = mlp
            batch_state.model_gathered_logprobs = mglp
            batch_state.model_encoded = me
            batch_state.model_latent = ml
            batch_state.y_gumbel = F.one_hot(y, len(self.vocab.itos)) + mprob - mprob.detach()
            batch_state.value_input = me

            if self.base_model is not None:
                with torch.no_grad():
                    bo, blp, bglp, be, bl = self.get_rl_tensors(self.base_model, x, y, latent_info, sources)
            else:
                bo, blp, bglp, be, bl = None, None, None, None, None

            batch_state.base_output = bo
            batch_state.base_logprobs = blp
            batch_state.base_gathered_logprobs = bglp
            batch_state.base_encoded = be
            batch_state.base_latent = bl


# Cell

class SupervisedCB(Callback):
    '''
    SupervisedCB - supervised training callback. When triggered,
    this callback grabs the top `percentile` of samples from the
    log and runs supervised training with the sampled data

    Inputs:

    - `agent Agent`: agent

    - `frequency int`: how often to run supervised training

    - `base_update float`: how much to update the baseline model
    after supervised training (if applicable)

    - `percentile int`: percentile (int value 1-100) of data
    to sample from the log

    - `lr float`: learning rate

    - `bs int`: batch size

    - `log_term str`: what term in the log to take the percentile of

    - `epochs int`: number of training epochs

    - `silent bool`: if training losses should be printed
    '''
    def __init__(self, agent, frequency, base_update, percentile,
                 lr, bs, log_term='rewards', epochs=1, silent=True):
        super().__init__('supervised', order=1000)
        self.agent = agent
        self.frequency = frequency
        self.base_update = base_update
        self.percentile = percentile
        self.lr = lr
        self.bs = bs
        self.log_term = log_term
        self.epochs = epochs
        self.silent = silent

    def after_batch(self):
        env = self.environment
        iterations = self.environment.log.iterations

        if iterations>0 and iterations%self.frequency==0:
            self.train_model()

    def train_model(self):
        env = self.environment
        df = env.log.df[['samples', self.log_term]]
        df = df[df[self.log_term]>np.percentile(df[self.log_term].values, self.percentile)]

        self.agent.update_dataset_from_inputs(df.samples.values)
        self.agent.train_supervised(self.bs, self.epochs, self.lr, silent=self.silent)

        if hasattr(self.agent, 'base_model'):
            if isinstance(self.agent.base_model, nn.Module):
                merge_models(self.agent.base_model, self.agent.model, alpha=self.base_update)

# Cell

class Rollback(Callback):
    '''
    Rollback - if `metric_name` falls (above/below) `target`, updates
    the main model's weights with the baseline model's weights

    Inputs:

    - `agent BaselineAgent`: agent

    - `metric_name str`: metric to track

    - `lookback int`: number of batches to look back. Also sets the
    maximum rollback frequency

    - `target float`: desired cutoff for `metric_name`

    - `alpha float`: during rollback, the main model weights are
    updated following `model = alpha*model + (1-alpha)*base_model`

    - `name str`: callback name

    - `mode str['greater', 'lesser']`: if greater, rollback is triggered by
    the metric going over `target`. If lesser, rollback is triggered by the
    metric falling below `target`
    '''
    def __init__(self, agent, metric_name, lookback, target, alpha, name, mode='greater'):
        super().__init__(name=name)
        self.agent = agent
        assert self.agent.base_model is not None
        self.metric_name = metric_name
        self.lookback = lookback
        self.target = target
        self.alpha = alpha
        self.mode = mode
        self.last_rollback = 0

    def after_batch(self):
        log = self.environment.log
        current_value = np.array(log.metrics[self.metric_name][-self.lookback:]).mean()

        if self.mode == 'greater':
            condition = current_val > self.target
        else:
            condition = current_val < self.target

        if condition and self.last_rollback <= 0:
            merge_models(self.agent.model, self.agent.base_model, self.alpha)
            self.last_rollback = self.lookback

        self.last_rollback -= 1


# Cell

class RetrainRollback(Callback):
    '''
    RetrainRollback - triggers supervised training if
    `metric_name` falls (above/below) `target`

    Inputs:

    - `agent BaselineAgent`: agent

    - `metric_name str`: metric to track

    - `log_term str`: what term in the log to take the percentile of

    - `lookback int`: number of batches to look back. Also sets the
    maximum rollback frequency

    - `target float`: desired cutoff for `metric_name`

    - `percentile int`: percentile (1-100) of data to sample from the log

    - `lr float`: learning rate

    - `bs int`: batch size

    - `base_update float`: after supervised training, the weights
    of the baseline model are updated following
    `base_model = alpha*base_model + (1-alpha)*model`

    - `name str`: callback name

    - `mode str['greater', 'lesser']`: if greater, rollback is triggered by
    the metric going over `target`. If lesser, rollback is triggered by the
    metric falling below `target`

    - `silent bool`: if training losses should be printed
    '''
    def __init__(self, agent, metric_name, log_term, lookback, target,
                 percentile, lr, bs, base_update, name, mode='greater',
                 silent=False):
        super().__init__(name=name, order=1000)

        self.agent = agent
        self.metric_name = metric_name
        self.log_term = log_term
        self.lookback = lookback
        self.target = target
        self.percentile = percentile
        self.lr = lr
        self.bs = bs
        self.base_update = base_update
        self.mode = mode
        self.last_rollback = 0
        self.silent = silent

    def after_batch(self):
        log = self.environment.log
        current_value = np.array(log.metrics[self.metric_name][-self.lookback:]).mean()

        if self.mode == 'greater':
            condition = current_value > self.target
        else:
            condition = current_value < self.target

        if condition and self.last_rollback <= 0:
            self.train_model()
            self.last_rollback = self.lookback

        self.last_rollback -= 1


    def train_model(self):
        env = self.environment
        df = env.log.df
        df.drop_duplicates(subset='samples', inplace=True)

        metric_values = df[self.log_term]

        df = df[metric_values>np.percentile(metric_values, self.percentile)]

        self.agent.update_dataset_from_inputs(df.samples.values)
        self.agent.train_supervised(self.bs, 1, self.lr, silent=self.silent)

        merge_models(self.agent.base_model, self.agent.model, alpha=self.base_update)

# Cell

class ResetAndRetrain(Callback):
    '''
    ResetAndRetrain - with a set frequency, loads a
    file of saved weights and runs supervised training

    Inputs:

    - `agent BaselineAgent`: agent

    - `frequency int`: how often to run supervised training

    - `weight_fp str`: filepath to weights

    - `percentile int`: percentile (int value 1-100) of data
    to sample from the log

    - `lr float`: learning rate

    - `bs int`: batch size

    - `epochs int`: number of epochs to run

    - `log_term str`: what term in the log to take the percentile of

    - `sample_term str`: what log term contains the samples to train on

    - `silent bool`: if training losses should be printed
    '''
    def __init__(self, agent, frequency, weight_fp, percentile,
                 lr, bs, epochs, log_term='rewards', sample_term='samples',
                 silent=False):
        super().__init__(name='reset_retrain', order=1000)
        self.agent = agent
        self.frequency = frequency
        self.percentile = percentile
        self.lr = lr
        self.bs = bs
        self.epochs = epochs
        self.log_term = log_term
        self.sample_term = sample_term
        self.weight_fp = weight_fp
        self.silent = silent

    def after_batch(self):
        env = self.environment
        iterations = self.environment.log.iterations

        if iterations>0 and iterations%self.frequency==0:
            self.train_model()


    def train_model(self):
        env = self.environment
        df = env.log.df[[self.sample_term, self.log_term]]
        df = df[df[self.log_term]>np.percentile(df[self.log_term].values, self.percentile)]

        self.agent.model.load_state_dict(torch.load(self.weight_fp))

        self.agent.update_dataset_from_inputs(df[self.sample_term].values)
        self.agent.train_supervised(self.bs, self.epochs, self.lr, silent=self.silent)

        self.agent.base_model.load_state_dict(self.agent.model.state_dict())

# Cell

class MetricResetAndRetrain(Callback):
    '''
    MetricResetAndRetrain - loads a file of saved
    weights and runs supervised training if
    `metric_name` falls (above/below) `target`

    Inputs:

    - `agent BaselineAgent`: agent

    - `metric_name str`: metric to track

    - `lookback int`: number of batches to look back. Also sets the
    maximum rollback frequency

    - `target float`: desired cutoff for `metric_name`

    - `weight_fp str`: filepath to weights

    - `percentile int`: percentile (int value 1-100) of data
    to sample from the log

    - `lr float`: learning rate

    - `bs int`: batch size

    - `epochs int`: number of epochs to run

    - `log_term str`: what term in the log to take the percentile of

    - `sample_term str`: what log term contains the samples to train on

    - `mode str['greater', 'lesser']`: if greater, rollback is triggered by
    the metric going over `target`. If lesser, rollback is triggered by the
    metric falling below `target`

    - `silent bool`: if training losses should be printed
    '''
    def __init__(self, agent, metric_name, lookback, target,
                 weight_fp, percentile, lr, bs, epochs,
                 log_term='rewards', sample_term='samples',
                 mode='greater', silent=False):
        super().__init__(name='metric_retrain', order=1000)

        self.agent = agent
        self.metric_name = metric_name
        self.lookback = lookback
        self.target = target
        self.percentile = percentile
        self.lr = lr
        self.bs = bs
        self.epochs = epochs
        self.log_term = log_term
        self.sample_term = sample_term
        self.weight_fp = weight_fp
        self.silent = silent
        self.mode = mode
        self.last_rollback = 0

    def after_batch(self):
        log = self.environment.log
        current_value = np.array(log.metrics[self.metric_name][-self.lookback:]).mean()

        if self.mode == 'greater':
            condition = current_value > self.target
        else:
            condition = current_value < self.target

        if condition and self.last_rollback <= 0:
            self.train_model()
            self.last_rollback = self.lookback

        self.last_rollback -= 1


    def train_model(self):
        env = self.environment
        df = env.log.df[[self.sample_term, self.log_term]]
        df = df[df[self.log_term]>np.percentile(df[self.log_term].values, self.percentile)]

        self.agent.model.load_state_dict(torch.load(self.weight_fp))

        self.agent.update_dataset_from_inputs(df[self.sample_term].values)
        self.agent.train_supervised(self.bs, self.epochs, self.lr, silent=self.silent)

        self.agent.base_model.load_state_dict(self.agent.model.state_dict())

# Cell

class SaveAgentWeights(Callback):
    '''
    SaveAgentWeights - saves weights every `n_batches`.
    Weights are saved to `file_path/filename_iterations.pt`

    Inputs:

    - `file_path str`: directory to save weights in

    - `filename str`: base filename

    - `n_batches int`: how often to save weights

    - `agent Agent`: agent
    '''
    def __init__(self, file_path, filename, n_batches, agent):
        super().__init__(name='save_cb')

        self.file_path = file_path
        self.filename = filename

    def after_batch(self):
        env = self.environment
        iterations = log.iterations

        if iterations>0 and (n_batches%iterations)==0:
            filename = self.file_path + self.filename + f'_{iterations}.pt'
            agent.save_weights(filename)