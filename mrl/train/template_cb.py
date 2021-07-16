# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/15_template_callback.ipynb (unless otherwise specified).

__all__ = ['TemplateCallback', 'ContrastiveTemplate', 'SimilarityFunction', 'FPSimilarity']

# Cell

from .callback import *
from ..chem import *
from ..templates.all import *
from ..torch_imports import *
from ..torch_core import *

# Cell

class TemplateCallback(Callback):
    '''
    TemplateCallback - callback wrapper for `Template` class

    Inputs:

    - `template Template`: template to use

    - `sample_name str`: `BatchState` attribute to grab samples from

    - `weight float`: weight used to scale template soft filter score

    - `track bool`: if True, template results are added to the batch log
    and metric log

    - `prefilter bool`: if True, samples that fail hard filters in the
    template are removed

    - `do_filter bool`: controls if filtering is done at aall
    '''
    def __init__(self, template=None, sample_name='samples', weight=1.,
                 track=True, prefilter=True, do_filter=True):
        super().__init__(order=-1)
        self.template = template
        self.track = track
        self.name = 'template'
        self.prefilter = prefilter
        self.weight = weight
        self.sample_name = sample_name
        self.do_filter = do_filter

    def setup(self):
        if self.track:
            log = self.environment.log
            log.add_metric(self.name)
            log.add_log(self.name)
            log.add_metric(f'valid')

            if isinstance(self.template, BlockTemplate):
                log.add_log('samples_fused')

    def filter_buffer(self):
        if self.do_filter:
            env = self.environment
            buffer = env.buffer
            if buffer.buffer:
                buffer.buffer = self.standardize(buffer.buffer)
                valids = self.filter_sequences(buffer.buffer, return_array=True)
                buffer._filter_buffer(valids)

    def filter_batch(self):
        valid = 1.
        env = self.environment
        if self.do_filter:
            batch_state = env.batch_state

            samples = batch_state[self.sample_name]
            samples = self.standardize(samples)
            batch_state[self.sample_name] = samples

            valids = self.filter_sequences(samples, return_array=True)

            self._filter_batch(valids)

            valid = valids.mean()

        if self.track:
            env.log.update_metric('valid', valid)


    def compute_reward(self):
        env = self.environment
        state = env.batch_state

        if isinstance(self.template, BlockTemplate):
            outputs = self.template.recurse_fragments(state[self.sample_name])
            rewards = np.array([i[3] for i in outputs])
            state[self.sample_name+'_fused'] = [i[1] for i in outputs]

        elif self.template is not None:
            rewards = np.array(self.template.eval_mols(state[self.sample_name]))

        else:
            rewards = np.array([0.]*len(state[self.sample_name]))

        hps = self.get_hps(state[self.sample_name])
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
    '''
    ContrastiveTemplate -  contrasttive callback wrapper for `Template` class

    Inputs:

    - `similarity_function SimilarityFunction`: evaluates similarity between
    source and targe samples

    - `sample_name str`: `BatchState` attribute to grab samples from

    - `max_score Optional[float]`: maximum template reward. If given, will be
    used to scale contrastive scores

    - `template Template`: template to use

    - `weight float`: weight used to scale template soft filter score

    - `track bool`: if True, template results are added to the batch log
    and metric log

    - `prefilter bool`: if True, samples that fail hard filters in the
    template are removed

    - `do_filter bool`: controls if filtering is done at aall
    '''
    def __init__(self, similarity_function, sample_name='samples',
                 max_score=None, template=None,
                 weight=1., track=True, prefilter=True, do_filter=True):
        super().__init__(template=template,
                         sample_name=sample_name,
                         weight=weight,
                         track=track,
                         prefilter=prefilter,
                         do_filter=do_filter)

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

        samples = state[self.sample_name]
        source_samples = [i[0] for i in samples]
        target_samples = [i[1] for i in samples]
        hps = self.get_hps(samples)

        if self.template is not None:

            if isinstance(self.template, BlockTemplate):
                source_outputs = self.template.recurse_fragments(source_samples)
                target_outputs = self.template.recurse_fragments(target_samples)
                state[self.sample_name+'_fused'] = [(source_outputs[i][1], target_outputs[i][1])
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
            rewards = np.array([0.]*len(state[self.sample_name]))

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

class SimilarityFunction():
    '''
    SimilarityFunction - compares similarity between source
    and target samples
    '''
    def score(self, source_smiles, target_smiles):
        return [0. for i in source_smiles]

    def bools(self, source_smiles, target_smiles):
        return [True for i in source_smiles]

class FPSimilarity(SimilarityFunction):
    '''
    FPSimilarity - computes paired sample similarity using fingerprint
    similarity

    Inputs:

    - `fp_function Callable`: Fingerprint function, ie `ECFP6`. Should
    return a fingerprint when called

    - `similarity_function Callable`: fingerprint similarity function,
    ie `tanimoto`

    - `min_sim float`: minimum similarity between paired samples

    - `max_sim float`: maximum similarity between samples

    - `passscore float`: score for passing samples

    - `failscore float`: score for failling compounds

    - `soft_min Optional[float]`: if given, this value is used as
    the minimum similarity cutoff during scoring but not for filtering

    - `soft_max Optional[float]`: if given, this value is used as
    the maximum similarity cutoff during scoring but not for filtering
    '''
    def __init__(self, fp_function, similarity_function, min_sim, max_sim,
                 passscore, failscore, soft_min=None, soft_max=None):
        self.fp_function = fp_function
        self.similarity_function = similarity_function
        self.min_sim = min_sim
        self.max_sim = max_sim
        self.passscore = passscore
        self.failscore = failscore
        self.soft_min = soft_min
        self.soft_max = soft_max

    def get_sims(self, source_smiles, target_smiles):
        source_fps = maybe_parallel(self.fp_function, source_smiles)
        target_fps = maybe_parallel(self.fp_function, target_smiles)

        sims = np.array([self.similarity_function(source_fps[i], [target_fps[i]])[0]
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