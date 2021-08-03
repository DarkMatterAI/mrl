# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/20_environment.ipynb (unless otherwise specified).

__all__ = ['Environment']

# Cell

from ..imports import *
from .callback import *
from .log import *
from .buffer import *
from .template_cb import TemplateCallback
from ..templates.all import *

# Cell

class Environment():
    '''
    Environment - Environment for training `agent`

    Inputs:

    - `agent Optional[Agent]`: agent to train

    - `template_cb Optional[TemplateCallback]`: template callback

    - `samplers Optional[list[Sampler]]`: any sampler callbacks (can be any amount)

    - `rewards Optional[list[RewardCallback]]`: any reward callbacks

    - `losses Optional[list[LossCallback]]`: any loss callbacks

    - `cbs Optional[list[Callback]]`: any other callbacks

    - `buffer_p_batch Optional[float]`: percentage of each batch that
    should come from the buffer. If None, value is inferred from
    `p_batch` values in `samplers`

    - `log Optional[Log]`: custom log. If None, standard `Log` is used
    '''
    def __init__(self, agent=None, template_cb=None, samplers=None, rewards=None, losses=None,
                 cbs=None, buffer=None, log=None):

        if samplers is None:
            samplers = []

        if rewards is None:
            rewards = []

        if losses is None:
            losses = []

        if cbs is None:
            cbs = []

        if log is None:
            log = Log()

        self.agent = agent
        self.template_cb = template_cb
        self.samplers = samplers
        self.rewards = rewards
        self.losses = losses
        self.cbs = []
        self.verbose = False

        if buffer is None:
            buffer_p_batch = 1.
            for samp in samplers:
                buffer_p_batch -= samp.p_batch

            buffer = Buffer(buffer_p_batch)

        self.buffer = buffer
        self.batch_state = BatchState()
        self.log = log

        agent_cb = [self.agent] if self.agent is not None else []
        all_cbs = agent_cb + [self.template_cb] + self.samplers + self.rewards
        all_cbs += self.losses + cbs + [self.buffer] + [self.log]

        self.register_cbs(all_cbs)
        self('setup')

    def __call__(self, event):
        if self.verbose:
            print(event)
        for cb in self.cbs:
            cb(event)

    def register_cb(self, cb):
        if isinstance(cb, type):
            cb = cb()

        if cb is not None:
            cb.environment = self
            setattr(self, cb.name, cb)
            self.cbs.append(cb)

    def register_cbs(self, cbs):
        for cb in cbs:
            self.register_cb(cb)

        self.sort_cbs()

    def remove_cb(self, cb):
        cb.environment = None
        cb.batch_state = None
        if hasattr(self, cb.name):
            delattr(self, cb.name)

        if cb in self.cbs:
            self.cbs.remove(cb)

    def remove_cbs(self, cbs):
        for cb in cbs:
            self.remove_cb(cb)

        self.sort_cbs()

    def sort_cbs(self):
        if self.cbs:
            self.cbs = sorted(self.cbs, key=lambda x: x.order)

    def build_buffer(self):
        '''
        build_buffer

        If the current buffer length is less than
        the current batch size, this functiton
        runs the `build_buffer`, `filter_buffer`,
        and `after_build_buffer` events
        '''
        start = time.time()
        if (len(self.buffer) < self.bs) or self.bs==-1 or self.log.iterations==0:
            self('build_buffer')
            self('filter_buffer')
            self('after_build_buffer')
        end = time.time() - start
        self.log.timelog['build_buffer'].append(end)

    def sample_batch(self):
        '''
        sample_batch

        This function runs:
        - `before_batch`
        - `sample_batch`
        - `before_filter_batch`
        - `filter_batch`
        - `after_sample`
        '''
        start = time.time()
        self.batch_state = BatchState()
        self('before_batch')
        self('sample_batch')
        self('before_filter_batch')
        self('filter_batch')
        self('after_sample')
        end = time.time() - start
        self.log.timelog['sample_batch'].append(end)

    def get_model_outputs(self):
        '''
        get_model_outputs

        This function runs:
        - `get_model_outputs`
        - `after_get_model_outputs`
        '''
        start = time.time()
        self('get_model_outputs')
        self('after_get_model_outputs')
        end = time.time() - start
        self.log.timelog['get_model_outputs'].append(end)

    def compute_reward(self):
        '''
        compute_reward

        This function runs:
        - `before_compute_reward`
        - `compute_reward`
        - `after_compute_reward`
        - `reward_modification`
        - `after_reward_modification`
        '''
        start = time.time()
        self('before_compute_reward')
        self('compute_reward')
        self('after_compute_reward')
        self('reward_modification')
        self('after_reward_modification')
        end = time.time() - start
        self.log.timelog['compute_reward'].append(end)

    def compute_loss(self):
        '''
        compute_loss

        This function runs:
        - `compute_loss`
        - `zero_grad`
        - `before_step`
        - `step`
        '''
        start = time.time()
        self('compute_loss')
        loss = self.batch_state.loss
        self('zero_grad')
        loss.backward()
        self('before_step')
        self('step')
        end = time.time() - start

        self.log.timelog['compute_loss'].append(end)

    def after_batch(self):
        '''
        after_batch

        This function runs:
        - `after_batch`
        '''
        start = time.time()
        self('after_batch')
        end = time.time() - start
        self.log.timelog['after_batch'].append(end)

    def step(self):
        '''
        One step
        '''
        self.build_buffer()
        self.sample_batch()
        self.compute_reward()
        self.get_model_outputs()
        self.compute_loss()
        self.after_batch()

    def fit(self, bs, sl, iters, report, cbs=None, verbose=False):
        '''
        fit - runs the fit cycle

        Inputs:

        - `bs int`: batch size

        - `sl int`: max sample length

        - `iters int`: number of batches to train

        - `report int`: report batch stats every `report` batches

        - `cbs Optional[list[Callback]]`: optional callbacks
        for the fit loop

        - `verbose Bool`: if True, prints event calls
        '''
        self.verbose = verbose
        if cbs is None:
            cbs = []
        self.register_cbs(cbs)
        self.bs = bs
        self.sl = sl
        self.report = report
        mb = master_bar(range(1))
        self.log.pbar = mb
        self.log.report = report
        self('before_train')
        for _ in mb:
            for step in progress_bar(range(iters), parent=mb):
                self.step()
#                 self.build_buffer()
#                 self.sample_batch()
#                 self.compute_reward()
#                 self.get_model_outputs()
#                 self.compute_loss()
#                 self.after_batch()

        self('after_train')
        self.remove_cbs(cbs)
        self.verbose = False

    def plot_event_times(self, event):
        event_times = [i.event_timelog[event] for i in self.cbs]
        labels = [i.name for i in self.cbs]

        fig, ax = plt.subplots(figsize=(8,6))
        ax.stackplot(np.arange(len(event_times[0])), *event_times, labels=labels);
        ax.legend(loc='upper left');
