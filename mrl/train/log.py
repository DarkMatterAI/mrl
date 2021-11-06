# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/13_logging.ipynb (unless otherwise specified).

__all__ = ['Log', 'log_to_df', 'StatsCallback', 'MaxCallback', 'MinCallback', 'MeanCallback', 'PercentileCallback',
           'SaveLogDF']

# Cell

from ..imports import *
from ..core import *
from ..torch_imports import *
from ..torch_core import *
from .callback import *

# Cell

class Log(Callback):
    def __init__(self):
        super().__init__(name='log', order=100)

        self.pbar = None
        self.iterations = 0
        self.metrics = {}

        self.batch_log = {}
        self.timelog = defaultdict(list)

        self.report = 1
        self.unique_samples = {}

        self.add_metric('rewards')
        self.add_metric('rewards_final')
        self.add_metric('new')
        self.add_metric('diversity')
        self.add_metric('bs')

        self.add_log('samples')
        self.add_log('sources')
        self.add_log('rewards')
        self.add_log('rewards_final')

    def setup(self):
        self.df = pd.DataFrame(self.batch_log)

    def before_train(self):
        cols = ['iterations'] + list(self.metrics.keys())
        if self.pbar is None:
            print('\t'.join(cols))
        else:
            self.pbar.write(cols, table=True)

    def add_metric(self, name):
        if not name in self.metrics.keys():
            self.metrics[name] = []

    def add_log(self, name):
        if not name in self.batch_log.keys():
            self.batch_log[name] = []

    def update_metric(self, name, value):
        self.metrics[name].append(value)

    def after_sample(self):
        env = self.environment
        batch_state = env.batch_state
        samples = batch_state.samples
        batch_state.rewards = to_device(torch.zeros(len(samples)))

        new = np.array([not i in self.unique_samples for i in samples])

        self.update_metric('new', new.mean())

        if len(samples)>0:
            diversity = len(set(samples))/len(samples)
        else:
            diversity = 0.
        self.environment.log.update_metric('diversity', diversity)

        self.environment.log.update_metric('bs', len(batch_state.samples))

    def after_compute_reward(self):
        env = self.environment
        batch_state = env.batch_state
        samples = batch_state.samples
        rewards = batch_state.rewards
        batch_state.rewards_final = rewards.clone().detach()

        rewards = rewards.detach().cpu().numpy()

        self.update_metric('rewards', rewards.mean())

        for i in range(len(samples)):
            if not samples[i] in self.unique_samples:
                self.unique_samples[samples[i]] = rewards[i]

    def after_reward_modification(self):
        env = self.environment
        rewards = env.batch_state.rewards_final.detach().cpu().numpy()
        self.update_metric('rewards_final', rewards.mean())


    def update_log(self):
        env = self.environment
        batch_state = env.batch_state
        samples = batch_state.samples
        update_dict = {}

        for key in self.batch_log.keys():
            items = batch_state[key]
            if isinstance(items, torch.Tensor):
                items = items.detach().cpu().numpy()
            self.batch_log[key].append(items)
            update_dict[key] = items

        new_df = pd.DataFrame(update_dict)
        self.add_data(new_df)
#         repeats = new_df.samples.isin(self.df.samples)
#         new_df = new_df[~repeats]

#         self.df = self.df.append(new_df, ignore_index=True)

        if self.iterations%10==0 and self.iterations>0:
            self.df.drop_duplicates(subset='samples', inplace=True)

    def add_data(self, new_df):
        repeats = new_df.samples.isin(self.df.samples)
        new_df = new_df[~repeats]
        self.df = self.df.append(new_df, ignore_index=True)

    def report_batch(self):
        outputs = [f'{self.iterations}']
        if self.iterations%self.report==0:

            for k,v in self.metrics.items():
                val = v[-1]

                if type(val)==int:
                    val = f'{val}'
                elif type(val)==str:
                    val = val
                else:
                    val = f'{val:.3f}'

                outputs.append(val)

            if self.pbar is None:
                print('\t'.join(outputs))
            else:
                self.pbar.write(outputs, table=True)

        self.iterations += 1

    def after_batch(self):
        self.update_log()
        self.report_batch()

    def get_df(self):
        return log_to_df(self.batch_log)

    def plot_metrics(self, cols=4, smooth=True):
        self.plot_dict(self.metrics, cols=cols, smooth=smooth)

    def plot_timelog(self, cols=4, smooth=True):
        self.plot_dict(self.timelog, cols=cols, smooth=smooth)


# Cell

def log_to_df(log, keys=None):
    batch = 0
    output_dict = defaultdict(list)

    if keys is None:
        keys = list(log.keys())

    items = log[keys[0]]
    for item in items:
        output_dict['batch'] += [batch]*len(item)
        batch += 1

    for key in keys:
        output_dict[key] = flatten_list_of_lists(log[key])

    return pd.DataFrame(output_dict)

# Cell

class StatsCallback(Callback):
    '''
    StatsCallback - base class for callbacks related to calculating
    stats from batches

    Inputs:

    - `batch_attribute str`: attribute to grab from the log

    - `grabname Optional[str]`: if passed, the `batch_attribute` values
    will be subset for those where `source==grabname`

    - `include_buffer bool`: if True, values sourced from the buffer
    that match `grabname` will be included

    - `name str`: callback name

    - `order int`: callback order
    '''
    def __init__(self, batch_attribute, grabname=None, include_buffer=True,
                     name='stats', order=20):
        super().__init__(name=name, order=order)

        self.grabname = grabname
        self.batch_attribute = batch_attribute
        self.include_buffer = include_buffer

    def get_values(self):
        batch_state = self.environment.batch_state
        sources = np.array(batch_state.sources)

        if self.include_buffer:
            sources = np.array([i.replace('_buffer', '') for i in sources])

        values = batch_state[self.batch_attribute]

        if self.grabname is not None:
            source_mask = sources==self.grabname
            if source_mask.sum()>0:
                values = values[source_mask]
            else:
                values = np.array([0.])

        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        return values

# Cell

class MaxCallback(StatsCallback):
    '''
    MaxCallback - adds a metric tracking the maximum of
    `batch_attribute`, subset by `grabname`, printed every
    batch report

    Inputs:

    - `batch_attribute str`: attribute to grab from the log

    - `grabname Optional[str]`: if passed, the `batch_attribute` values
    will be subset for those where `source==grabname`

    - `include_buffer bool`: if True, values sourced from the buffer
    that match `grabname` will be included
    '''
    def __init__(self, batch_attribute, grabname, include_buffer=True):

        if grabname is None:
            name = f'{batch_attribute}_max'
        else:
            name = f'{batch_attribute}_{grabname}_max'

        super().__init__(batch_attribute, grabname,
                         include_buffer=include_buffer, name=name)


    def setup(self):
        log = self.environment.log
        log.add_metric(self.name)

    def after_compute_reward(self):

        values = self.get_values()
        self.environment.log.update_metric(self.name, values.max())

# Cell

class MinCallback(StatsCallback):
    '''
    MinCallback - adds a metric tracking the minimum of
    `batch_attribute`, subset by `grabname`, printed every
    batch report

    Inputs:

    - `batch_attribute str`: attribute to grab from the log

    - `grabname Optional[str]`: if passed, the `batch_attribute` values
    will be subset for those where `source==grabname`

    - `include_buffer bool`: if True, values sourced from the buffer
    that match `grabname` will be included
    '''
    def __init__(self, batch_attribute, grabname, include_buffer=True):

        if grabname is None:
            name = f'{batch_attribute}_min'
        else:
            name = f'{batch_attribute}_{grabname}_min'

        super().__init__(batch_attribute, grabname,
                         include_buffer=include_buffer, name=name)


    def setup(self):
        log = self.environment.log
        log.add_metric(self.name)

    def after_compute_reward(self):

        values = self.get_values()
        self.environment.log.update_metric(self.name, values.min())

# Cell

class MeanCallback(StatsCallback):
    '''
    MeanCallback - adds a metric tracking the mean of
    `batch_attribute`, subset by `grabname`, printed every
    batch report

    Inputs:

    - `batch_attribute str`: attribute to grab from the log

    - `grabname Optional[str]`: if passed, the `batch_attribute` values
    will be subset for those where `source==grabname`

    - `include_buffer bool`: if True, values sourced from the buffer
    that match `grabname` will be included
    '''
    def __init__(self, batch_attribute, grabname, include_buffer=True):

        if grabname is None:
            name = f'{batch_attribute}_mean'
        else:
            name = f'{batch_attribute}_{grabname}_mean'

        super().__init__(batch_attribute, grabname,
                         include_buffer=include_buffer, name=name)


    def setup(self):
        log = self.environment.log
        log.add_metric(self.name)

    def after_compute_reward(self):

        values = self.get_values()
        self.environment.log.update_metric(self.name, values.mean())

# Cell

class PercentileCallback(StatsCallback):
    '''
    PercentileCallback - adds a metric tracking the `percentile`
    percentile value of `batch_attribute`, subset by `grabname`,
    printed every batch report

    Inputs:

    - `batch_attribute str`: attribute to grab from the log

    - `grabname Optional[str]`: if passed, the `batch_attribute` values
    will be subset for those where `source==grabname`

    - `percentile str`: what percentile value to use

    - `include_buffer bool`: if True, values sourced from the buffer
    that match `grabname` will be included
    '''
    def __init__(self, batch_attribute, grabname, percentile, include_buffer=True):

        if grabname is None:
            name = f'{batch_attribute}_p{percentile}'
        else:
            name = f'{batch_attribute}_{grabname}_p{percentile}'

        super().__init__(batch_attribute, grabname,
                         include_buffer=include_buffer, name=name)
        self.percentile = percentile

    def setup(self):
        log = self.environment.log
        log.add_metric(self.name)

    def after_compute_reward(self):

        values = self.get_values()
        self.environment.log.update_metric(self.name, np.percentile(values, self.percentile))


# Cell

class SaveLogDF(Callback):
    '''
    SaveLogDF - periodically saves the Log
    dataframe during training

    Inputs:

    - `frequency int`: how often to save

    - `save_path str`: directory to save
    files to
    '''
    def __init__(self, frequency, save_path):
        super().__init__(name='save_log')
        self.frequency = frequency
        self.save_path = save_path

    def after_batch(self):
        log = self.environment.log
        if (log.iterations%self.frequency)==0 and log.iterations>0:
            log.df.to_csv(f'{self.save_path}/log_df_{log.iterations}.csv', index=False)
