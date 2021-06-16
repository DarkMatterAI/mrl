# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_template.filters.ipynb (unless otherwise specified).

__all__ = ['ScoreFunction', 'NoScore', 'PassThroughScore', 'ModifiedScore', 'ConstantScore', 'WeightedPropertyScore',
           'PropertyFunctionScore', 'LinearDecayScore', 'Filter', 'ValidityFilter', 'SingleCompoundFilter',
           'PropertyFilter', 'MolWtFilter', 'HBDFilter', 'HBAFilter', 'TPSAFilter', 'RotBondFilter', 'SP3Filter',
           'LogPFilter', 'RingFilter', 'HeteroatomFilter', 'AromaticRingFilter', 'HeavyAtomsFilter', 'MRFilter',
           'ChargeFilter', 'TotalAtomFilter', 'QEDFilter', 'SAFilter', 'StructureFilter', 'ExclusionFilter', 'FPFilter']

# Cell
from ..imports import *
from ..core import *
from ..chem import *

# Cell

class ScoreFunction():
    "Base score function"
    def __call__(self, property_output, criteria_output):
        pass

class NoScore(ScoreFunction):
    "Pass through for no score"
    def __call__(self, property_output, criteria_output):
        return 0.

class PassThroughScore(ScoreFunction):
    "Pass through for property_output"
    def __call__(self, property_output, criteria_output):
        return property_output

class ModifiedScore(ScoreFunction):
    "Base class for scores where property_output is modified by some function"
    def __init__(self, fail_score=0.):
        self.fail_score = float(fail_score)

    def __call__(self, property_output, criteria_output):

        if not criteria_output and self.fail_score is not None:
            output = self.fail_score
        else:
            output = self.compute_score(property_output)

        return output

    def compute_score(self, property_output):
        raise NotImplementedError

class ConstantScore(ModifiedScore):
    "Returns pass_score if criteria_output, else fail_score"
    def __init__(self, pass_score, fail_score):
        super().__init__(fail_score)
        self.pass_score = float(pass_score)

    def compute_score(self, property_output):
        return self.pass_score

class WeightedPropertyScore(ModifiedScore):
    "Returns weight*property_output if criteria_output, else fail_score"
    def __init__(self, weight, fail_score=0.):
        super().__init__(fail_score)
        self.weight = weight

    def compute_score(self, property_output):
        return property_output*self.weight

class PropertyFunctionScore(ModifiedScore):
    "Returns output `function(property_output)`"
    def __init__(self, function, fail_score=0.):
        super().__init__(fail_score)
        self.function = function

    def compute_score(self, property_output):
        return self.function(property_output)


class LinearDecayScore(ScoreFunction):
    '''
    LinearDecayScore - score with linear decay. `low_start<low_end<high_start<high_end`

    Returns `pass_score` if `criteria_output=True` and
    `low_end<=property_output<=high_start`. If `low_start<=property_output<=low_end` or
    `high_start<=property_output<=high_end`, the score is a linear interpolation between `pass_score`
    and `fail_score`. Otherwise, returns `fail_score`.

    One of `low_end`, `high_start` must be not None.

    If one of `low_end`, `high_start` is None, the corresponding bound is ignored

    if `low_start` or `high_end` is None, the score immediately drops to `fail_score`

    '''
    # low_start < low_end < high_start < high_end
    def __init__(self, pass_score, low_start, low_end,
                 high_start, high_end, fail_score=0.):

        self.pass_score = float(pass_score)
        self.fail_score = float(fail_score)

        self.low_start = low_start
        self.high_start = high_start

        self.low_end = low_end
        self.high_end = high_end

        assert (self.low_end is not None) or (self.high_start is not None), ("One of (low_end, high_start) "
                                                                             "must not be None")

    def check_bound(self, property_output, bound, boundtype):
        if bound is None:
            output = True
        else:
            if boundtype=='low':
                output = property_output>=bound
            else:
                output = property_output<=bound

        return output

    def __call__(self, property_output, criteria_output):

        if criteria_output:

            low_bound = self.check_bound(property_output, self.low_end, 'low')
            high_bound = self.check_bound(property_output, self.high_start, 'high')

            if low_bound and high_bound:
                # in main range
                output = self.pass_score

            elif low_bound:
                # above high start:
                high_end = self.check_bound(property_output, self.high_end, 'high')

                if high_end:
                    # between high_start and high_end
                    if (self.high_start is not None) and (self.high_end is not None):
                        fraction = (property_output - self.high_start)/(self.high_end - self.high_start)
                        output = self.pass_score*(1-fraction) + self.fail_score*fraction
                    else:
                        output = self.fail_score

                else:
                    output = self.fail_score

            else:
                # below low end
                low_start = self.check_bound(property_output, self.low_start, 'low')

                if low_start:
                    # between low_start and low_end
                    if (self.low_start is not None) and (self.low_end is not None):
                        fraction = (property_output - self.low_start)/(self.low_end - self.low_start)
                        output = self.pass_score*fraction + self.fail_score*(1-fraction)
                    else:
                        output = self.fail_score

                else:
                    output = self.fail_score

        else:
            output = self.fail_score

        return output


# Cell

class Filter():
    '''
    Filter - base filter function class

    Inputs:

        `score` - one of (None, int, float, ScoreFunction), see `set_score`

        `name` - (str, None), filter name used for repr

        `fail_score` - (float, int), used in `set_score` if `score_function` is (int, float)

        `mode` - (str), `smile` or `protein`, determines how inputs are converted to Mol objects
    '''
    def __init__(self, score=None, name=None, fail_score=0., mode='smile'):
        self.score_function = self.set_score(score, fail_score)
        self.name = name
        self.priority = 0
        self.mode = mode

    def set_score(self, score_function, fail_score):
        if score_function is None:
            score = NoScore()
        elif type(score_function) in (int, float):
            score = ConstantScore(score_function, fail_score=fail_score)
        elif isinstance(score_function, ScoreFunction):
            score = score_function
        else:
            raise ValueError('Invalid score_function input, must be one of (None, int, float, ScoreFunction)')

        return score

    def __call__(self, mol, with_score=False):
        output = maybe_parallel(self.eval_mol, mol, with_score=with_score)
        return output

    def to_mol(self, input):
        if self.mode=='smile':
            mol = to_mol(input)
        elif self.mode=='protein':
            mol = to_protein(input)
        else:
            raise ValueError("`self.mode` must be one of `['smile', 'protein']`")

        return mol

    def to_string(self, input):
        if self.mode=='smile':
            string = to_smile(input)
        elif self.mode=='protein':
            string = to_sequence(input)
        else:
            raise ValueError("`self.mode` must be one of `['smile', 'protein']`")

        return string

    def eval_mol(self, mol, with_score=False):
        '''
        eval_mol - evaluates `Mol` based on `property_function`.
            if `with_score=True`, returns the output of `score_function`, else
            returns the output of `property_function
        '''
        mol = self.to_mol(mol)
        property_output = self.property_function(mol)
        criteria_output = self.criteria_function(property_output)

        if with_score:
            output = self.score_function(property_output, criteria_output)
        else:
            output = criteria_output

        return output

    def property_function(self, mol):
        raise NotImplementedError

    def criteria_function(self, property_output):
        raise NotImplementedError

    def __repr__(self):
        if self.name is not None:
            output = f'{self.name}'
        else:
            output = 'Unnamed Filter'

        return output

# Cell
class ValidityFilter(Filter):
    '''
    ValidityFilter - checks to see if a given `Mol` is a valid compound

    Inputs:

        `score` - one of (None, int, float, ScoreFunction), see `set_score`

        `name` - (str, None), filter name used for repr

        `fail_score` - (float, int), used in `set_score` if `score_function` is (int, float)

        `mode` - (str), `smile` or `protein`, determines how inputs are converted to Mol objects
    '''
    def __init__(self, score=None, name=None, fail_score=0., mode='smile'):
        if name is None:
            name = 'Vaidity Filter'

        super().__init__(score=score, name=name, fail_score=fail_score, mode=mode)
        self.priority=2

    def property_function(self, mol):
        mol = self.to_mol(mol)
        return mol

    def criteria_function(self, property_output):
        return property_output is not None and property_output.GetNumAtoms() > 0

class SingleCompoundFilter(Filter):
    '''
    SingleCompoundFilter - checks to see if a given `Mol` is a single compound
    '''
    def __init__(self, score=None, name=None, fail_score=0.):
        if name is None:
            name = 'Single Compound Filter'

        super().__init__(score, name, fail_score=fail_score)
        self.priority=1

    def property_function(self, mol):
        smile = self.to_string(mol)
        return smile

    def criteria_function(self, property_output):
        return not '.' in property_output

# Cell

class PropertyFilter(Filter):
    '''
    PropertyFilter - filters mols based on `mol_function`

    Inputs:

        `mol_function` - any function that takes as input a `Mol` object and returns a single numeric value

        `min_val` - (None, int, float), inclusive lower bound for filter (ignored if None)

        `max_val` - (None, int, float), inclusive upper bound for filter (ignored if None)

        `score` - one of (None, int, float, ScoreFunction), see `FilterFunction.set_score`

        `name` - (str, None), filter name used for repr

        `fail_score` - (float, int), used in `set_score` if `score_function` is (int, float)

        `mode` - (str), `smile` or `protein`, determines how inputs are converted to Mol objects
    '''
    def __init__(self, mol_function, min_val=None, max_val=None, score=None,
                 fail_score=0., name=None, mode='smile'):

        self.mol_function = mol_function
        self.min_val = min_val
        self.max_val = max_val

        if name is None:
            name = mol_function.__name__

        super().__init__(score, name, fail_score=fail_score, mode=mode)

    def property_function(self, mol):
        return self.mol_function(mol)

    def criteria_function(self, property_output):
        lower_bound = (property_output>=self.min_val) if self.min_val is not None else True
        upper_bound = (property_output<=self.max_val) if self.max_val is not None else True
        output = lower_bound and upper_bound

        return output

    def __repr__(self):
        output = f'{self.name}' + f' ({self.min_val}, {self.max_val})'
        return output

class MolWtFilter(PropertyFilter):
    "Molecular weight filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(molwt, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class HBDFilter(PropertyFilter):
    "Hydrogen bond donor filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(hbd, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class HBAFilter(PropertyFilter):
    "Hydrogen bond acceptor filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(hba, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class TPSAFilter(PropertyFilter):
    "TPSA filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(tpsa, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class RotBondFilter(PropertyFilter):
    "Rotatable bond filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(rotbond, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class SP3Filter(PropertyFilter):
    "Fractioon sp3 filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(fsp3, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class LogPFilter(PropertyFilter):
    "LogP filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(logp, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class RingFilter(PropertyFilter):
    "Ring filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(rings, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class HeteroatomFilter(PropertyFilter):
    "Heteroatom filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(heteroatoms, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class AromaticRingFilter(PropertyFilter):
    "Aromatic ring filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(aromaticrings, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class HeavyAtomsFilter(PropertyFilter):
    "Number of heavy atoms filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(heavy_atoms, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class MRFilter(PropertyFilter):
    "Molar refractivity of atoms filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(molar_refractivity, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class ChargeFilter(PropertyFilter):
    "Formal charge of atoms filter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(formal_charge, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class TotalAtomFilter(PropertyFilter):
    "Total number of atoms filter (incudes H)"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(all_atoms, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class QEDFilter(PropertyFilter):
    "Total number of atoms filter (incudes H)"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(qed, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

class SAFilter(PropertyFilter):
    "SA Score fillter"
    def __init__(self, min_val, max_val, score=None, name=None, **kwargs):
        super().__init__(sa_score, min_val=min_val, max_val=max_val, score=score, name=name, **kwargs)

# Cell

class StructureFilter(Filter):
    '''
    StructureFilter - filters mols based on structures in `smarts`

    Inputs:

        `smarts` - (list, SmartsCatalog), list of smarts strings for filtering or `SmartsCatalog`

        `exclude` - if True, filter returns `False` when a structure match is found

        `criteria` - ('any', 'all'), match criteria (match any filter, match all filters)

        `score` - one of (None, int, float, ScoreFunction), see `FilterFunction.set_score`

        `name` - (str, None), filter name used for repr

        `fail_score` - (float, int), used in `set_score` if `score_function` is (int, float)
    '''
    def __init__(self, smarts, exclude=True, criteria='any', score=None, name=None, fail_score=0.):

        self.catalog = self.get_catalog(smarts)
        self.exclude = exclude
        self.criteria = criteria
        assert self.criteria in ('any', 'all'), "`criteria` must be one of ('any', 'all')"

        if name is None:
            name = f'Structure filter, criteria: {criteria}, exclude: {exclude}'

        super().__init__(score, name, fail_score=fail_score)

    def property_function(self, mol):
        return self.catalog(mol, criteria=self.criteria)

    def criteria_function(self, property_output):
        if not is_container(property_output):
            property_output = [property_output]

        if self.criteria=='any':
            output = any(property_output)
        else:
            output = all(property_output)

        if self.exclude:
            output = not output

        return output

    def get_catalog(self, smarts):
        if isinstance(smarts, Catalog):
            smarts = smarts
        else:
            smarts = SmartsCatalog(smarts)
        return smarts

class ExclusionFilter(StructureFilter):
    '''
    ExclusionFilter - excludes mols with substructure matches to `smarts`

    Inputs:

        `smarts` - (list, SmartsCatalog), list of smarts strings for filtering or `SmartsCatalog`

        `criteria` - ('any', 'all'), match criteria (match any filter, match all filters)

        `score` - one of (None, int, float, ScoreFunction), see `FilterFunction.set_score`

        `name` - (str, None), filter name used for repr

        `fail_score` - (float, int), used in `set_score` if `score_function` is (int, float)
    '''
    def __init__(self, smarts, criteria='any', score=None, name=None, fail_score=0.):

        if name is None:
            name = f'Excusion filter, criteria: {criteria}'

        super().__init__(smarts, exclude=True, criteria=criteria,
                         score=score, name=name, fail_score=fail_score)

# Cell

class FPFilter(Filter):
    '''
    FPFilter - filters mols based on fingerprint similarity to `reference_smiles`

    Inputs:

        `reference_smiles` - (list), list of smiles or `Mol` objects for comparison

        `fp_type` - fingerprint function. see `FP`

        `fp_metric` - fingerprint similarity metric. see `FP`

        `criteria` - ('any', 'all'), match criteria (match any reference, match all references)

        `fp_thresh` - float, fingerprint similarity cutoff for defining a match

        `name` - (str, None), filter name used for repr

        `fail_score` - (float, int), used in `set_score` if `score_function` is (int, float)

        `score` - one of (None, int, float, ScoreFunction), see `FilterFunction.set_score`
    '''
    def __init__(self, reference_fps, fp_type, fp_metric, criteria='any',
                fp_thresh=0., score=None, name=None, fail_score=0.):

        self.reference_fps = reference_fps
        self.fp = FP()
        self.fp_type = fp_type
        self.fp_metric = fp_metric
        self.array_type = self.fp._np_or_rd(reference_fps)
        self.get_fp = partial(self.fp.get_fingerprint, fp_type=self.fp_type, output_type=self.array_type)
        self.get_similaity = partial(self.fp.fingerprint_similarity,
                                     fps2=self.reference_fps, metric=fp_metric)
        self.criteria = criteria
        self.fp_thresh = fp_thresh

        if name is None:
            name = f'Fingerprint Filter, {fp_type}, {fp_metric}, {len(reference_fps)} references'

        super().__init__(score, name, fail_score=fail_score)

    def property_function(self, mol):
        fp = self.get_fp(mol)
        similarity = self.get_similaity(fp)
        return similarity

    def criteria_function(self, property_output):
        property_output = property_output>=self.fp_thresh

        if not is_container(property_output):
            property_output = [property_output]

        if self.criteria=='any':
            output = any(property_output)
        else:
            output = all(property_output)

        return output

    @classmethod
    def from_smiles(cls, reference_smiles, fp_type='ECFP6', fp_metric='tanimoto',
                    criteria='any', fp_thresh=0., score=None, name=None, fail_score=0,):
        '''
        creates FPFilter from `reference_smiles`

        `reference_smiles` can be a list of smiles or a list of `Mols`
        '''
        reference_fps = get_fingerprint(reference_smiles, fp_type=fp_type)
        return cls(reference_fps, fp_type, fp_metric,
                   criteria=criteria, fp_thresh=fp_thresh, score=score,
                   name=name, fail_score=fail_score)
