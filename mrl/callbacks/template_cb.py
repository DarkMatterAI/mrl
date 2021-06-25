# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/15_template_callback.ipynb (unless otherwise specified).

__all__ = ['TemplateCallback', 'ContrastiveTemplate', 'FPSimilarity']

# Cell

from .core import *
from ..chem import *
from ..templates import *
from ..torch_imports import *
from ..torch_core import *

# Cell

class TemplateCallback(Callback):
    def __init__(self, template=None, weight=1., track=True, prefilter=True):
        super().__init__(order=-1)
        self.template = template
        self.track = track
        self.name = 'template'
        self.prefilter = prefilter
        self.weight = weight

    def setup(self):
        if self.track:
            log = self.environment.log
            log.add_metric(self.name)
            log.add_log(self.name)
            log.add_metric(f'valid')

            if isinstance(self.template, BlockTemplate):
                log.add_log('samples_fused')

    def after_build_buffer(self):
        env = self.environment
        buffer = env.buffer
        if buffer.buffer:
            buffer.buffer = self.standardize(buffer.buffer)
            buffer.buffer = self.filter_sequences(buffer.buffer)

    def after_sample(self):
        env = self.environment
        batch_state = env.batch_state

        samples = batch_state.samples
        samples = self.standardize(samples)
        batch_state.samples = samples

        sources = np.array(batch_state.sources)
        valids = self.filter_sequences(samples, return_array=True)

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

        if self.track:
            env.log.update_metric('valid', valids.mean())

    def compute_reward(self):
        env = self.environment
        state = env.batch_state

        if isinstance(self.template, BlockTemplate):
            outputs = self.template.recurse_fragments(state.samples)
            rewards = np.array([i[3] for i in outputs])
            state.samples_fused = [i[1] for i in outputs]

        elif self.template is not None:
            rewards = np.array(self.template.eval_mols(state.samples))

        else:
            rewards = np.array([0.]*len(state.samples))

        hps = self.get_hps(state.samples)
        state[self.name] = rewards
        rewards = rewards*self.weight

        if self.track:
            env.log.update_metric(self.name, rewards.mean())

        state.template_passes = hps
        state.rewards += to_device(torch.from_numpy(rewards).float())

    def get_hps(self, sequences):
        if self.template is not None:
            hps = np.array(self.template(sequences))
        else:
            hps = np.array([True]*len(sequences))

        return hps

    def filter_sequences(self, sequences, return_array=False):
        if self.prefilter:
            passes = self.get_hps(sequences)
        else:
            passes = self.validate(sequences)

        if return_array:
            output = passes
        else:
            output  = [sequences[i] for i in range(len(sequences)) if passes[i]]
        return output

    def standardize(self, sequences):
        if self.template is not None:
            sequences = self.template.standardize(sequences)

        return sequences

    def validate(self, sequences):
        if self.template is not None:
            valid = np.array(self.template.validate(sequences))
        else:
            valid = np.array([True]*len(sequences))

        return valid

# Cell

class ContrastiveTemplate(TemplateCallback):
    def __init__(self, similarity_function, max_score=None, template=None,
                 weight=1., track=True, prefilter=True):
        super().__init__(template=template, weight=weight, track=track, prefilter=prefilter)
        self.similarity_function = similarity_function
        self.max_score = max_score

    def setup(self):
        if self.track:
            log = self.environment.log
            log.add_metric(self.name)
            log.add_metric(self.name+'_temp')
            log.add_metric(self.name+'_sim')
            log.add_metric(f'valid')

            log.add_log(self.name)
            log.add_log(self.name+'_temp')
            log.add_log(self.name+'_sim')

            if isinstance(self.template, BlockTemplate):
                log.add_log('samples_fused')

    def compute_reward(self):
        env = self.environment
        state = env.batch_state

        samples = state.samples
        source_samples = [i[0] for i in samples]
        target_samples = [i[1] for i in samples]
        hps = self.get_hps(samples)

        if self.template is not None:

            if isinstance(self.template, BlockTemplate):
                source_outputs = self.template.recurse_fragments(source_samples)
                target_outputs = self.template.recurse_fragments(target_samples)
                state.samples_fused = [(source_outputs[i][1], target_outputs[i][1])
                                      for i in range(len(source_outputs))]

                source_rewards = np.array([i[3] for i in source_outputs])
                target_rewards = np.array([i[3] for i in target_outputs])

            else:
                source_rewards = np.array(self.template.eval_mols(source_samples))
                target_rewards = np.array(self.template.eval_mols(target_samples))

            rewards = target_rewards - source_rewards
            if self.max_score is not None:
                rewards = rewards / (self.max_score-source_rewards)

        else:
            rewards = np.array([0.]*len(state.samples))

        sim_scores = self.similarity_function.score(source_samples, target_samples)

        state.template = rewards
        state.template_sim = sim_scores

        full_rewards = rewards + sim_scores
        full_rewards = full_rewards*self.weight

        if self.track:
            env.log.update_metric(self.name, full_rewards.mean())
            env.log.update_metric(self.name+'_temp', rewards.mean())
            env.log.update_metric(self.name+'_sim', sim_scores.mean())

        state[self.name] = full_rewards
        state[self.name+'_temp'] = rewards
        state[self.name+'_sim'] = sim_scores

        state.template_passes = hps
        state.rewards += to_device(torch.from_numpy(full_rewards).float())

    def standardize(self, sequences):
        if self.template is not None:
            sources = self.template.standardize([i[0] for i in sequences])
            targets = self.template.standardize([i[1] for i in sequences])
            sequences = [(sources[i], targets[i]) for i in range(len(sources))]

        return sequences

    def get_hps(self, sequences):

        if type(sequences[0])==str:
            hps = super().get_hps(sequences)
        else:
            source_sequences = [i[0] for i in sequences]
            target_sequences = [i[1] for i in sequences]
            s_hps = super().get_hps(source_sequences)
            t_hps = super().get_hps(target_sequences)
            sim_bools = self.similarity_function.bools(source_sequences, target_sequences)
            hps = s_hps*t_hps*sim_bools

        return hps

    def validate(self, sequences):
        if type(sequences[0])==str:
            valid = super().validate(sequences)
        else:
            s_val = super().validate([i[0] for i in sequences])
            t_val = super().validate([i[1] for i in sequences])
            valid = s_val*t_val

        return valid


# Cell

class FPSimilarity():
    def __init__(self, fp_function, distance_function, min_sim, max_sim,
                 passscore, failscore, soft_min=None, soft_max=None):
        self.fp_function = fp_function
        self.distance_function = distance_function
        self.min_sim = min_sim
        self.max_sim = max_sim
        self.passscore = passscore
        self.failscore = failscore
        self.soft_min = soft_min
        self.soft_max = soft_max

    def get_sims(self, source_smiles, target_smiles):
        source_fps = [failsafe_fp(i, self.fp_function) for i in source_smiles]
        target_fps = [failsafe_fp(i, self.fp_function) for i in target_smiles]

        sims = np.array([self.distance_function(source_fps[i], [target_fps[i]])[0]
                 for i in range(len(source_smiles))])
        return sims

    def score(self, source_smiles, target_smiles, sims=None):
        if sims is None:
            sims = self.get_sims(source_smiles, target_smiles)

        min_sim = self.min_sim if self.soft_min is None else self.soft_min
        max_sim = self.max_sim if self.soft_max is None else self.soft_max
        bools = (min_sim<sims) & (sims<max_sim)

        return bools*self.passscore + (~bools)*self.failscore

    def bools(self, source_smiles, target_smiles, sims=None):
        if sims is None:
            sims = self.get_sims(source_smiles, target_smiles)

        bools = (self.min_sim<sims) & (sims<self.max_sim)
        return bools