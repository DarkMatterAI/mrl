# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/14_buffer.ipynb (unless otherwise specified).

__all__ = ['Buffer']

# Cell

from .core import *

# Cell

class Buffer(Callback):
    def __init__(self, p_total):
        super().__init__(name='buffer', order=0)

        self.buffer = []
        self.p_total = p_total
        self.buffer_valid = []

    def __len__(self):
        return len(self.buffer)

    def add(self, item):

        if is_container(item):
            for i in item:
                self.add(i)
        else:
            self.buffer.append(item)

    def setup(self):
        log = self.environment.log
        log.add_metric(f'diversity')
        log.add_metric(f'valid')
        log.add_metric(f'bs')

    def sample(self, n):

        idxs = np.random.choice(np.arange(len(self.buffer)), min(n, len(self.buffer)),
                                replace=False)
        batch = [self.buffer[i] for i in idxs]
        for idx in sorted(idxs, reverse=True):
            self.buffer.pop(idx)

        return batch

    def after_sample(self):
        env = self.environment
        batch_state = env.batch_state

        template_cb = env.template_cb
        samples = batch_state.samples
        samples = template_cb.standardize(samples)
        batch_state.samples = samples

        sources = np.array(batch_state.sources)
        valids = template_cb.filter_sequences(samples, return_array=True)

        if valids.mean()<1.:
            filtered_samples = [samples[i] for i in range(len(samples)) if valids[i]]
            filtered_sources = [sources[i] for i in range(len(sources)) if valids[i]]
            filtered_latent_data = {}

            for source,latent_idxs in batch_state.latent_data.items():
                valid_subset = valids[sources==source]
                latent_filtered = latent_idxs[valid_subset]
                filtered_latent_data[source] = latent_filtered

            batch_state.samples = filtered_samples
            batch_state.sources = filtered_sources
            batch_state.latent_data = filtered_latent_data


        diversity = len(set(batch_state.samples))/len(batch_state.samples)
        self.environment.log.update_metric('diversity', diversity)
        self.environment.log.update_metric('valid', valids.mean())
        self.environment.log.update_metric('bs', len(batch_state.samples))

    def after_build_buffer(self):
        template_cb = self.environment.template_cb
        if self.buffer:
            len1 = len(self.buffer)
            self.buffer = template_cb.standardize(self.buffer)
            self.buffer = list(set(self.buffer))
            self.buffer = template_cb.filter_sequences(self.buffer)
            self.buffer_valid.append(len(self.buffer)/len1)

    def sample_batch(self):
        env = self.environment
        batch_state = env.batch_state

        bs = int(env.bs * self.p_total)
        if bs>0:
            sample = self.sample(bs)
            batch_state.samples += sample
            batch_state.sources += ['buffer']*len(sample)