# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/13_logging.ipynb (unless otherwise specified).

__all__ = ['log_to_df', 'Log']

# Cell

from .core import *
from ..torch_imports import *

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

class Log(Callback):
    def __init__(self):
        super().__init__(name='log', order=100)

        self.pbar = None
        self.iterations = 0
        self.metrics = {}

        self.log = {}
        self.timelog = defaultdict(list)

        self.report = 1
        self.unique_samples = set()

        self.add_metric('rewards')
        self.add_log('samples')
        self.add_log('sources')
        self.add_log('rewards')

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
        if not name in self.log.keys():
            self.log[name] = []

    def update_metric(self, name, value):
        self.metrics[name].append(value)

    def update_log(self):
        env = self.environment
        batch_state = env.batch_state
        samples = batch_state.samples
        self.unique_samples.update(set(samples))

        for key in self.log.keys():
            try:
                items = batch_state[key]
                if isinstance(items, torch.Tensor):
                    items = items.detach().cpu().numpy()
                self.log[key].append(items)
            except:
                pass

    def report_batch(self):
        outputs = [f'{self.iterations}']
        if self.iterations%self.report==0:

            for k,v in self.metrics.items():
                val = v[-1]

                if type(val)==int:
                    val = f'{val}'
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
        return log_to_df(self.log)

