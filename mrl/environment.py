# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/20_environment.ipynb (unless otherwise specified).

__all__ = ['Environment']

# Cell

from .imports import *
from .callbacks import *
from .callbacks.template_cb import TemplateCallback

# Cell

class Environment():
    def __init__(self, agent_cb, template_cb=None, samplers=None, reward_cbs=None, loss_cbs=None,
                 cbs=None, buffer_p_batch=None, log=None):

        if samplers is None:
            samplers = []

        if reward_cbs is None:
            reward_cbs = []

        if loss_cbs is None:
            loss_cbs = []

        if cbs is None:
            cbs = []

        if log is None:
            log = Log()

        self.agent_cb = agent_cb
        self.template_cb = template_cb if template_cb is not None else TemplateCallback()
        self.samplers = samplers
        self.reward_cbs = reward_cbs
        self.loss_cbs = loss_cbs
        self.cbs = []

        if buffer_p_batch is None:
            buffer_p_batch = 1.
            for samp in samplers:
                buffer_p_batch -= samp.p_batch

        self.buffer = Buffer(buffer_p_batch)
        self.batch_state = BatchState()
        self.log = log

        all_cbs = [self.agent_cb] + [self.template_cb] + self.samplers + self.reward_cbs
        all_cbs += self.loss_cbs + cbs + [self.buffer] + [self.log]
        all_cbs = sorted(all_cbs, key=lambda x: x.order)

        self.register_cbs(all_cbs)
        self('setup')

    def __call__(self, event):
        for cb in self.cbs:
            cb(event)

    def register_cb(self, cb):
        if isinstance(cb, type):
            cb = cb()
        cb.environment = self
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def register_cbs(self, cbs):
        for cb in cbs:
            self.register_cb(cb)

    def remove_cb(self, cb):
        cb.environment = None
        cb.batch_state = None
        if hasattr(self, cb.name):
            delattr(self, cb.name)

        if cb in self.cbs:
            self.cbs.remove(cb)

    def remove_cbs(self, cbs):
        for cb in cbs:
            self.remove_cb(ccb)

    def build_buffer(self):
        start = time.time()
        if (len(self.buffer) < self.bs):
            self('build_buffer')
            self('filter_buffer')
            self('after_build_buffer')
        end = time.time() - start
        self.log.timelog['build_buffer'].append(end)

    def sample_batch(self):
        start = time.time()
        self.batch_state = BatchState()
        self('before_batch')
        self('sample_batch')
        self('filter_batch')
        self('after_sample')
        end = time.time() - start
        self.log.timelog['sample_batch'].append(end)

    def get_model_outputs(self):
        start = time.time()
        self('get_model_outputs')
        end = time.time() - start
        self.log.timelog['get_model_outputs'].append(end)

    def compute_reward(self):
        start = time.time()
        self('before_compute_reward')
        self('compute_reward')
        rewards = self.batch_state.rewards

        self.log.update_metric('rewards', rewards.mean().detach().cpu().numpy())

        self('after_compute_reward')
        end = time.time() - start
        self.log.timelog['compute_reward'].append(end)

    def compute_loss(self):
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
        start = time.time()
        self('after_batch')
        end = time.time() - start
        self.log.timelog['after_batch'].append(end)

    def fit(self, bs, sl, iters, report, cbs=None):
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
                self.build_buffer()
                self.sample_batch()
                self.compute_reward()
                self.get_model_outputs()
                self.compute_loss()
                self.after_batch()

        self('after_train')
        self.remove_cbs(cbs)

    def plot_event_times(self, event):
        event_times = [i.event_timelog[event] for i in self.cbs]
        labels = [i.name for i in self.cbs]

        fig, ax = plt.subplots(figsize=(8,6))
        ax.stackplot(np.arange(len(event_times[0])), *event_times, labels=labels);
        ax.legend(loc='upper left');
