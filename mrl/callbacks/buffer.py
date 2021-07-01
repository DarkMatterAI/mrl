# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/14_buffer.ipynb (unless otherwise specified).

__all__ = ['Buffer']

# Cell

from ..imports import *
from .core import *

# Cell

class Buffer(Callback):
    def __init__(self, p_total):
        super().__init__(name='buffer', order=0)

        self.buffer = []
        self.buffer_sources = []
        self.p_total = p_total

    def __len__(self):
        return len(self.buffer)

    def add(self, item, name=''):

        if type(item)==list:
            for i in item:
                self.add(i, name=name)
        else:
            self.buffer.append(item)
            self.buffer_sources.append(name+'_buffer')

    def sample(self, n):

        idxs = np.random.choice(np.arange(len(self.buffer)), min(n, len(self.buffer)),
                                replace=False)
        batch = [self.buffer[i] for i in idxs]
        sources = [self.buffer_sources[i] for i in idxs]
        for idx in sorted(idxs, reverse=True):
            self.buffer.pop(idx)
            self.buffer_sources.pop(idx)

        return batch, sources

    def _filter_buffer(self, valids):

        self.buffer = [self.buffer[i] for i in range(len(self.buffer)) if valids[i]]
        self.buffer_sources = [self.buffer_sources[i]
                               for i in range(len(self.buffer_sources)) if valids[i]]

    def filter_buffer(self):
        if self.buffer:
            seen = set()
            unique = []
            for item in self.buffer:
                if item in seen:
                    unique.append(False)
                else:
                    seen.add(item)
                    unique.append(True)

            self._filter_buffer(np.array(unique))

    def sample_batch(self):
        env = self.environment
        batch_state = env.batch_state

        bs = int(env.bs * self.p_total)
        if bs>0:
            sample, sources = self.sample(bs)
            batch_state.samples += sample
            batch_state.sources += sources

    def filter_batch(self):
        env = self.environment
        batch_state = env.batch_state
        samples = batch_state.samples

        unique_samples = set()
        unique = []

        for sample in samples:
            if sample in unique_samples:
                unique.append(False)
            else:
                unique_samples.add(sample)
                unique.append(True)

        unique = np.array(unique)
        self._filter_batch(unique)
