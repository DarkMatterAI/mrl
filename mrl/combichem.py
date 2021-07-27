# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/23_combichem.ipynb (unless otherwise specified).

__all__ = ['Crossover', 'FragmentCrossover', 'Mutator', 'SmartsMutator', 'ChangeAtom', 'AppendAtomSingle',
           'AppendAtomsDouble', 'AppendAtomsTriple', 'AppendAtom', 'DeleteAtom', 'ChangeBond', 'InsertAtomSingle',
           'InsertAtomDouble', 'InsertAtomTriple', 'InsertAtom', 'AddRing', 'AllSmarts', 'AppendRgroupMutator',
           'EnumerateHeterocycleMutator', 'ShuffleNitrogen', 'ContractAtom', 'SelfiesMutator', 'SelfiesInsert',
           'SelfiesReplace', 'SelfiesRemove', 'MutatorCollection', 'CombiChem']

# Cell

from .imports import *
from .core import *
from .chem import *
from rdkit import Chem
from rdkit.Chem import EnumerateHeterocycles
import selfies as sf
from .torch_imports import *

# Cell

class Crossover():
    '''
    Crossover - base class for crossover events.
    To create custom crossovers, subclass `Crossover`
    and implement the `Crossover.crossover` method

    When called, `Crossover` is passed a list of `Mol`
    objects. The crossover operation randomly generates
    molecular pairs, and sends those pairs to `Crossover.crossover`
    '''
    def __init__(self, name='crossover'):
        self.name = name

    def __call__(self, mols):
        '''
        Crossover.__call__

        Inputs:

        `mols list[Chem.Mol]`: list of mol objects

        Returns:

        `list[str]`: list of smiles strings
        '''
        mols = to_mols(mols)
        shuffle_idxs = np.random.choice(range(len(mols)), len(mols), replace=False)
        pairs = [(mols[i], mols[shuffle_idxs[i]]) for i in range(len(mols))]
        outputs = maybe_parallel(self.crossover, pairs)
        return flatten_list_of_lists(outputs)

    def crossover(self, mol_pair):
        '''
        crossover - performs crossover operation

        Inputs:

        `mol_pair list[Chem.Mol, Chem.Mol]`: list of two
        Mol objects
        '''
        raise NotImplementedError

# Cell

class FragmentCrossover(Crossover):
    '''
    FragmentCrossover - crossover based on
    molecular fragmentation.

    Each Mol is fragmented into a set of
    `(scaffold, rgroup)` pairs by cutting
    single bonds in the molecule.

    During crossover, molecular pairs are
    merged following `scaffold1 + rgroup2`

    Inputs:

    `full_crossover bool`: if True, all
    `scaffold, rgroup` combinations are generated

    `name str`: crossover name
    '''
    def __init__(self, full_crossover=False, name='fragment crossover'):
        super().__init__(name)
        self.full_crossover = full_crossover

    def crossover(self, mol_pair):
        mol1, mol2 = mol_pair
        cores1, rgroups1 = self.split_fragments(self.fragment(to_mol(mol1)))
        cores2, rgroups2 = self.split_fragments(self.fragment(to_mol(mol2)))

        outputs = self.merge_groups(cores1, rgroups2)
        outputs += self.merge_groups(cores2, rgroups1)
        return outputs

    def merge_groups(self, cores, rgroups):
        random.shuffle(cores)
        random.shuffle(rgroups)
        if self.full_crossover:
            inputs = list(itertools.product(cores, rgroups))
            inputs = ['.'.join(i) for i in inputs]
        else:
            inputs = [cores[i]+'.'+rgroups[i] for i in range(min(len(cores), len(rgroups)))]
        fused = [fuse_on_atom_mapping(i) for i in inputs]
        return fused

    def fragment(self, mol):
        fragments = [i[1] for i in fragment_mol(mol, 1)]
        return fragments

    def split_fragments(self, fragments):
        cores = []
        rgroups = []

        for fragment in fragments:
            core, rgroup = self.split_fragment(fragment)
            cores.append(core)
            rgroups.append(rgroup)

        return cores, rgroups

    def split_fragment(self, fragment):
        f1, f2 = fragment.split('.')
        if len(f1)>len(f2):
            core = f1
            rgroup = f2
        else:
            core = f2
            rgroup = f1

        return core, rgroup

# Cell

class Mutator():
    '''
    Mutator - base class for mutations.
    To create custom mutations, subclass
    `Mutator` and implement `Mutator.mutate`
    '''
    def __init__(self, name=None):
        if name is None:
            name='Mutator'
        self.name = name
    def __call__(self, mols):
        if is_container(mols):
            mols = to_mols(mols)
            outputs = maybe_parallel(self.mutate, mols)
            outputs = flatten_list_of_lists(outputs)
        else:
            mols = to_mol(mols)
            outputs = self.mutate(mols)

        return outputs

    def mutate(self, mol):
        '''
        mutate - implement molecular mutation.

        Inputs:

        `mol Chem.Mol`: input mol

        Returns:

        `list[outputs]`
        '''
        raise NotImplementedError

    def __repr__(self):
        return self.name

# Cell

class SmartsMutator(Mutator):
    '''
    SmartsMutator - SMARTS reaction based
    mutator.

    Inputs:

    `smarts list[str]`: list of SMARTS reaction strings

    `name str`: mutator name
    '''
    def __init__(self, smarts, name=None):
        if name is None:
            name = 'Smarts Mutator'
        self.name = name

        self.smarts = smarts
        self.rxns = [smart_to_rxn(i) for i in self.smarts]
        for r in self.rxns:
            r.Initialize()

    def __add__(self, other, name=None):
        smarts = self.smarts + other.smarts
        if name is None:
            name = self.name + ' + ' + other.name

        return SmartsMutator(smarts, name=name)

    def mutate(self, mol):

        rxn_idxs = np.random.choice(range(len(self.rxns)), len(self.rxns), replace=False)
        products = []

        for idx in rxn_idxs:
            if self.rxns[idx].IsMoleculeReactant(mol):
                selected_rxn = self.rxns[idx]
                try:
                    products = [to_smile(i[0]) for i in selected_rxn.RunReactants([mol])]
                except:
                    pass
            if products:
                break

        return products

    def __repr__(self):
        r = f'{self.name}: {len(self.smarts)} Reactions'
        return r

# Cell

class ChangeAtom(SmartsMutator):
    '''
    ChangeAtom - SMARTS-based mutator that
    changes atom type without changing
    molecular structure

    Inputs:

    `atom_types Optional[list[str]]`: list of
    allowed atom types. Must be strings of
    atomic weights, ie `['6', '7', '8']`

    Default: `['6', '7', '8', '9', '15', '16', '17', '35']`
    '''
    def __init__(self, atom_types=None):
        if atom_types is None:
            atom_types = ['6', '7', '8', '9', '15', '16', '17', '35']

        smarts = []
        for a1 in atom_types:
            for a2 in atom_types:
                if not a1==a2:
                    smart = f'[#{a1}:1]>>[#{a2}:1]'
                    smarts.append(smart)

        super().__init__(smarts, name='Change Atom')

# Cell

class AppendAtomSingle(SmartsMutator):
    '''
    AppendAtomSingle - SMARTS-based mutator
    that appends an atom somewhere on the
    input structure with a single bond

    Inputs:

    `atom_types Optional[list[str]]`: list of
    allowed atom types. Must be strings of
    atomic symbols, ie `['C', 'N', 'O']`

    Default: `['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']`
    '''
    def __init__(self, atom_types=None):
        if atom_types is None:
            atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
        smarts = []
        for a in atom_types:
            smart = f'[*;!H0:1]>>[*:1]-{a}'
            smarts.append(smart)

        super().__init__(smarts, name='Append Atom Single Bond')

# Cell

class AppendAtomsDouble(SmartsMutator):
    '''
    AppendAtomsDouble - SMARTS-based mutator
    that appends an atom somewhere on the
    input structure with a double bond

    Inputs:

    `atom_types Optional[list[str]]`: list of
    allowed atom types. Must be strings of
    atomic symbols, ie `['C', 'N', 'O']`.
    Atom types must be compatible with
    forming a double bond

    Default: `['C', 'N', 'O', 'P', 'S']`
    '''
    def __init__(self, atom_types=None):
        if atom_types is None:
            atom_types = ['C', 'N', 'O', 'P', 'S']
        smarts = []
        for a in atom_types:
            smart = f'[*;!H0;!H1:1]>>[*:1]={a}'
            smarts.append(smart)

        super().__init__(smarts, name='Append Atom Double Bond')

# Cell

class AppendAtomsTriple(SmartsMutator):
    '''
    AppendAtomsTriple - SMARTS-based mutator
    that appends an atom somewhere on the
    input structure with a triple bond

    Inputs:

    `atom_types Optional[list[str]]`: list of
    allowed atom types. Must be strings of
    atomic symbols, ie `['C', 'N']`.
    Atom types must be compatible with
    forming a triple bond

    Default: `['C', 'N']`
    '''
    def __init__(self, atom_types=None):
        if atom_types is None:
            atom_types = ['C', 'N']
        smarts = []
        for a in atom_types:
            smart = f'[*;H3:1]>>[*:1]#{a}'
            smarts.append(smart)

        super().__init__(smarts, name='Append Atom Triple Bond')

# Cell

class AppendAtom(SmartsMutator):
    '''
    AppendAtom - SMARTS-based mutator
    that appends an atom somewhere on the
    input structure.

    Combines `AppendAtomSingle`,
    `AppendAtomsDouble` and `AppendAtomsTriple`
    '''
    def __init__(self):
        smarts = AppendAtomSingle().smarts
        smarts += AppendAtomsDouble().smarts
        smarts += AppendAtomsTriple().smarts
        super().__init__(smarts, name='Append Atom')

# Cell

class DeleteAtom(SmartsMutator):
    '''
    DeleteAtom - SMARTS-based mutator
    that randomly deletes an atom from
    the input structure
    '''
    def __init__(self):
        smarts = [
            '[*:1]~[D1]>>[*:1]',
            '[*:1]~[D2]~[*:2]>>[*:1]-[*:2]',
            '[*:1]~[D3](~[*;!H0:2])~[*:3]>>[*:1]-[*:2]-[*:3]',
            '[*:1]~[D4](~[*;!H0:2])(~[*;!H0:3])~[*:4]>>[*:1]-[*:2]-[*:3]-[*:4]',
            '[*:1]~[D4](~[*;!H0;!H1:2])(~[*:3])~[*:4]>>[*:1]-[*:2](-[*:3])-[*:4]'
        ]
        super().__init__(smarts, name='Delete Atom')

# Cell

class ChangeBond(SmartsMutator):
    '''
    ChangeBond - SMARTS-based mutator
    that randomly changes a bond in the
    input structure
    '''
    def __init__(self):
        smarts = [
            '[*:1]@[*:2]>>([*:1].[*:2])',
            '[*:1]!-[*:2]>>[*:1]-[*:2]',
            '[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]',
            '[*:1]#[*:2]>>[*:1]=[*:2]',
            '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]',
        ]
        super().__init__(smarts, name='Change Bond')

# Cell

class InsertAtomSingle(SmartsMutator):
    '''
    InsertAtomSingle - SMARTS-based mutator
    that randomly inserts an atom into
    the input structure with single bonds

    Inputs:

    `atom_types Optional[list[str]]`: list of
    allowed atom types. Must be strings of
    atomic symbols, ie `['C', 'N', 'O']`

    Default: `['C', 'N', 'O', 'P', 'S']`
    '''
    def __init__(self, atom_types=None):
        if atom_types is None:
            atom_types = ['C', 'N', 'O', 'P', 'S']
        smarts = []
        for a in atom_types:
            smart = f'[*:1]~[*:2]>>[*:1]{a}[*:2]'
            smarts.append(smart)

        super().__init__(smarts, name='Insert Atom Single')

# Cell

class InsertAtomDouble(SmartsMutator):
    '''
    InsertAtomDouble - SMARTS-based mutator
    that randomly inserts an atom into
    the input structure with a double bond

    Inputs:

    `atom_types Optional[list[str]]`: list of
    allowed atom types. Must be strings of
    atomic symbols, ie `['C', 'N', 'O']`.
    Atom types must be compatible with
    forming a double bond

    Default: `['C', 'N', 'P', 'S']`
    '''
    def __init__(self, atom_types=None):
        if atom_types is None:
            atom_types = ['C', 'N', 'P', 'S']
        smarts = []
        for a in atom_types:
            smart1 = f'[*;!H0:1]~[*:2]>>[*:1]={a}-[*:2]'
            smart2 = f'[*;!H0:1]~[*:2]>>[*:1]-{a}=[*:2]'
            smarts.append(smart1)
            smarts.append(smart2)

        super().__init__(smarts, name='Insert Atom Double')

# Cell

class InsertAtomTriple(SmartsMutator):
    '''
    InsertAtomTriple - SMARTS-based mutator
    that randomly inserts an atom into
    the input structure with a triple bond
    '''
    def __init__(self):
        smarts = ['[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#C-[*:2]',
                  '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]-C#[*:2]']
        super().__init__(smarts, name='Insert Atom Triple')

# Cell

class InsertAtom(SmartsMutator):
    '''
    InsertAtom - SMARTS-based mutator
    that randomly inserts an atom into
    the input structure.

    Combines `InsertAtomSingle`,
    `InsertAtomDouble` and `InsertAtomTriple`
    '''
    def __init__(self):
        smarts = InsertAtomSingle().smarts
        smarts += InsertAtomDouble().smarts
        smarts += InsertAtomTriple().smarts
        super().__init__(smarts, name='Insert Atom')

# Cell

class AddRing(SmartsMutator):
    '''
    AddRing - SMARTS-based mutator
    that randomly creates rings
    '''
    def __init__(self):
        smarts = [
        '[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1',
        '[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1',
        '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1',
        '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1',
        ]
        super().__init__(smarts, name='Change Bond')

# Cell

class AllSmarts(SmartsMutator):
    '''
    AllSmarts - SMARTS-based mutator
    that combines `ChangeAtom`,
    `AppendAtom`, `DeleteAtom`,
    `ChangeBond`, `InsertAtom`,
    and `AddRing`
    '''
    def __init__(self):
        smarts = ChangeAtom().smarts
        smarts += AppendAtom().smarts
        smarts += DeleteAtom().smarts
        smarts += ChangeBond().smarts
        smarts += InsertAtom().smarts
        smarts += AddRing().smarts

        super().__init__(smarts, name='Smarts Mutator')

# Cell

class AppendRgroupMutator(Mutator):
    '''
    AppendRgroupMutator - randomly
    appends r-groups to the input molecule

    Inputs:

    `rgroups list[str]`: list of rgroups. All
    rgroups should have a single wildcard (`*`) atom

    `name str`: mutator name
    '''
    def __init__(self, rgroups, name='Rgroup'):
        super().__init__(name)

        self.pattern = re.compile('\[\*:.]')

        for rgroup in rgroups:
            assert rgroup.count('*')==1, f"Rgroup {rgroup} should only have 1 wildcard"

        rgroups = [self.clean_rgroup(i) for i in rgroups]
        self.rgroups = rgroups

    def clean_rgroup(self, rgroup):
        matches = self.pattern.findall(rgroup)
        for match in matches:
            rgroup = rgroup.replace(match, '*')

        rgroup = rgroup.replace('*', '[*:1]')
        return rgroup

    def mutate(self, mol):
        smile = to_smile(mol)
        decorated = decorate_smile(smile, 1)
        decorated = [i.replace('*', '[*:1]') for i in decorated]
        pairs = [i+'.'+np.random.choice(self.rgroups) for i in decorated]
        fused = [fuse_on_atom_mapping(i) for i in pairs]
        return fused

    def __repr__(self):
        r = f'{self.name}: {len(self.smarts)} Reactions'
        return r

# Cell

class EnumerateHeterocycleMutator(Mutator):
    '''
    EnumerateHeterocycleMutator - mutates
    input molecule by enumerating nitrogens
    on heterocycles

    Inputs:

    `depth int`: number of recursive enumerations

    `name str`: mutator name
    '''
    def __init__(self, depth=None, name='enum heteroatoms'):
        super().__init__(name)
        self.depth = depth

    def mutate(self, mol):
        new_mols = list(EnumerateHeterocycles.EnumerateHeterocycles(mol, depth=self.depth))
        new_mols = [i for i in new_mols if i is not None]
        smiles = [to_smile(i) for i in new_mols]
        smiles = list(set(smiles))
        return smiles

# Cell

class ShuffleNitrogen(Mutator):
    '''
    ShuffleNitrogen - mutates input molecule
    by shuffling the positions of carbon and nitrogen
    atoms in the molecule

    Inputs:

    `n_shuffles int`: number of shuffled variants to
    generate

    `name str`: mutator name
    '''
    def __init__(self, n_shuffles, name='shuffle nitrogen'):
        super().__init__(name)
        self.n_shuffles = n_shuffles
        self.aromatic_cH = Chem.MolFromSmarts("[cH]")
        self.aromatic_n = Chem.MolFromSmarts('[n]')
        self.normal_c = Chem.MolFromSmarts("[C]")
        self.normal_n = Chem.MolFromSmarts('[N]')

    def mutate(self, mol):
        cs = [i[0] for i in mol.GetSubstructMatches(self.aromatic_cH)]
        cs += [i[0] for i in mol.GetSubstructMatches(self.normal_c)]
        ns = [i[0] for i in mol.GetSubstructMatches(self.aromatic_n)]
        ns += [i[0] for i in mol.GetSubstructMatches(self.normal_n)]

        nums = [6]*len(cs) + [7]*len(ns)
        atom_idxs = cs + ns
        smile = to_smile(mol)
        outputs = []

        for i in range(self.n_shuffles):
            mol = to_mol(smile)
            idxs = np.random.choice(range(len(nums)), len(nums), replace=False)
            shuffle_idxs = [atom_idxs[i] for i in idxs]
            for i, aidx in enumerate(shuffle_idxs):
                atom = mol.GetAtomWithIdx(aidx)
                atom.SetAtomicNum(nums[i])

            new_smile = to_smile(mol)
            if new_smile:
                outputs.append(new_smile)

        return outputs

# Cell

class ContractAtom(Mutator):
    '''
    ContractAtom - mutates input molecule by
    removing an atom with two bonds and joining
    the removed atoms neighbors with a single bond.

    ie `a-b-c -> a-c`

    Inputs:

    `include_rings bool`: if True, rings will be
    contracted

    `name str`: mutator name
    '''
    def __init__(self, include_rings=True, name='contract'):
        super().__init__(name)
        self.include_rings = include_rings

    def mutate(self, mol):
        idxs = []
        for atom in mol.GetAtoms():
            if (atom.IsInRing() and self.include_rings) or (not atom.IsInRing()):
                bonds = atom.GetBonds()
                if len(bonds)==2:
                    idxs.append(atom.GetIdx())

        outputs = []
        for idx in idxs:
            outputs.append(self.contract(mol, idx))

        outputs = [i for i in outputs if i is not None]
        outputs = list(set(outputs))
        return outputs

    def contract(self, mol, idx):

        new_mol = Chem.RWMol(mol)

        atom = new_mol.GetAtomWithIdx(idx)
        bonds = atom.GetBonds()

        try:
            neighbors = atom.GetNeighbors()
            n1 = neighbors[0].GetIdx()
            n2 = neighbors[1].GetIdx()
            new_mol.AddBond(n1, n2, order=Chem.rdchem.BondType.SINGLE)

            new_mol.RemoveAtom(idx)

            mol = new_mol.GetMol()
            Chem.SanitizeMol(mol)
            output = to_smile(mol)
        except:
            output = None
        return output

# Cell

class SelfiesMutator(Mutator):
    '''
    SelfiesMutator - base class for SELFIES
    based mutation

    Inputs:

    `n_augs int`: number of mutated versions
    to generate

    `name str`: mutator name
    '''
    def __init__(self, n_augs, name='selfies'):
        super().__init__(name)
        self.n_augs = n_augs
        self.alphabet = list(sf.get_semantic_robust_alphabet())

    def mutate(self, mol):
        smile = to_smile(mol)
        selfie = smile_to_selfie(smile)
        tokens = list(sf.split_selfies(selfie))
        outputs = []

        for i in range(self.n_augs):
            new_tokens = self.augment(list(tokens))
            new_tokens = ''.join(new_tokens)
            try:
                smile = selfie_to_smile(new_tokens)
                smile = canon_smile(smile)
                outputs.append(smile)
            except:
                pass

        return outputs

    def augment(self, tokens):
        raise NotImplementedError

# Cell

class SelfiesInsert(SelfiesMutator):
    '''
    SelfiesInsert - SELFIES insertion mutator.
    Randomly inserts a SELFIES token into
    the input compound

    Inputs:

    `n_augs int`: number of mutated versions
    to generate

    `name str`: mutator name
    '''
    def __init__(self, n_augs, name='selfies insert'):
        super().__init__(n_augs, name)

    def augment(self, tokens):
        idx = np.random.randint(len(tokens)+1)
        new_token = np.random.choice(self.alphabet)
        new_tokens = tokens[:idx] + [new_token] + tokens[idx:]
        return new_tokens

# Cell

class SelfiesReplace(SelfiesMutator):
    '''
    SelfiesReplace - SELFIES replacement mutator.
    Randomly replaces a SELFIES token in
    the input compound

    Inputs:

    `n_augs int`: number of mutated versions
    to generate

    `name str`: mutator name
    '''
    def __init__(self, n_augs, name='selfies replace'):
        super().__init__(n_augs, name)

    def augment(self, tokens):
        idx = np.random.choice(range(len(tokens)))
        new_token = np.random.choice(self.alphabet)
        tokens[idx] = new_token
        return tokens

# Cell

class SelfiesRemove(SelfiesMutator):
    '''
    SelfiesRemove - SELFIES removal mutator.
    Randomly removes a SELFIES token in
    the input compound

    Inputs:

    `n_augs int`: number of mutated versions
    to generate

    `name str`: mutator name
    '''
    def __init__(self, n_augs, name='selfies remove'):
        super().__init__(n_augs, name)

    def augment(self, tokens):
        idx = np.random.choice(range(len(tokens)))
        tokens.pop(idx)
        return tokens

# Cell

class MutatorCollection():
    '''
    MutatorCollection - orchestrates a
    set of `Mutator` classes. When called,
    randomly selects a mutator to apply
    to the input mol

    Inputs:

    `mutators list[Mutator]`: list of mutator
    objects

    `p_mutators Optional[list[float]]`: Optional
    list of probabilities for selecting
    each mutator. If None, a uniform distribution
    is applied
    '''
    def __init__(self, mutators, p_mutators=None):
        self.mutators = mutators
        if p_mutators is None:
            p_mutators = [1/len(mutators) for i in mutators]

        self.p_mutators = p_mutators
        self.timelog = defaultdict(list)

    def __call__(self, mols):
        mutated = flatten_list_of_lists(maybe_parallel(self.mutate, mols))
        return mutated

    def mutate(self, mol):
        if self.mutators is not None:
            m = np.random.choice(self.mutators, 1, p=self.p_mutators)[0]
            start = time.time()
            outputs = m(mol)
            end = time.time()
            self.timelog[m.name].append(end-start)
        else:
            outputs = []

        return outputs

# Cell

class CombiChem():
    '''
    CombiChem - class for running
    combichem operations

    Inputs:

    `mutator_collection Optional[MutatorCollection]`:
    Collection of mutations to use

    `crossovers Optional[list[Crossover]]`:
    list of `Crossover` objects

    `template Optional[Template]`: `Template` to
    control chemical space

    `rewards Optional[Reward]`: Rewards to
    score molecules

    `prune_percentile int[0,100]`: Percentile
    of compounds to keep during pruning

    `max_library_size Optional[int]`: Maximum
    library size after pruning

    `log bool`: If True, compounds generated by
    combichem are logged

    `p_explore float[0.,1.]`: Percentage of
    compounds below `prune_percentile` to keep
    '''
    def __init__(self,
                 mutator_collection=None,
                 crossovers=None,
                 template=None,
                 rewards=None,
                 prune_percentile=90,
                 max_library_size=None,
                 log=False,
                 p_explore=0.
                ):

        self.mutator_collection = mutator_collection

        self.crossovers = crossovers

        self.template = template

        self.rewards = rewards

        self.prune_percentile = prune_percentile
        self.max_library_size = max_library_size

        self.library = pd.DataFrame([], columns=['smiles', 'mols', 'score'])
        self.old_library = pd.DataFrame([], columns=['smiles', 'score'])
        self.log = log
        self.p_explore = p_explore
        self.timelog = defaultdict(list)

    def step(self):
        new_library = self.build_generation()
        new_library = self.clean_library(new_library)
        self.append_data(new_library)
        self.score_library()
        self.prune_library()

    def build_generation(self):
        start = time.time()

        mols = self.library.mols.values

        mutated = self.mutate(mols)

        t1 = time.time()
        self.timelog['mutate'].append(t1-start)

        crossovers = self.crossover(mols)

        t2 = time.time()
        self.timelog['crossover'].append(t2-t1)
        self.timelog['build_generation'].append(t2-start)

        new_library = list(set(mutated+crossovers))
        return new_library

    def clean_library(self, library):
        start = time.time()
        if self.template is not None:
            library, _ = self.template.screen_mols(library)
            library = [i[0] for i in library]
        library = maybe_parallel(canon_smile, library)
        library = list(set(library))
        end = time.time()
        self.timelog['clean_library'].append(end-start)
        return library

    def mutate(self, mols):
        if self.mutator_collection is not None:
            outputs = self.mutator_collection(mols)
        else:
            outputs = []

        return outputs

    def crossover(self, mols):
        outputs = []
        if self.crossovers is not None:
            for cx in self.crossovers:
                outputs += cx(mols)
        return outputs

    def score_library(self):
        start = time.time()
        to_score = self.library[self.library.score.isna()]
        rewards = np.zeros(to_score.shape[0])

        if self.rewards is not None:
            rewards = np.zeros(to_score.shape[0])
            for reward in self.rewards:
                r_iter = reward(to_score.smiles.values)
                if isinstance(r_iter, torch.Tensor):
                    r_iter = r_iter.detach().cpu()
                rewards = rewards + np.array(r_iter)

        if self.template is not None:
            rewards += np.array(self.template(to_score.smiles.values, 'soft'))

        self.library.loc[to_score.index, 'score'] = rewards

        self.library['score'] = self.library.score.map(lambda x: float(x))
        end = time.time()
        self.timelog['score_library'].append(start-end)

    def prune_library(self):
        start = time.time()
        cutoff = np.percentile(self.library.score.values, self.prune_percentile)

        n_cutoff = self.library[self.library.score >= cutoff].shape[0]

        if self.max_library_size is not None:
            n_cutoff = min(n_cutoff, self.max_library_size)

        if self.p_explore>0.:
            n_explore = int(self.p_explore*n_cutoff)
            n_cutoff = int((1-self.p_explore)*n_cutoff)

        idxs = self.library.score.nlargest(n_cutoff).index

        new_library = self.library.iloc[idxs]

        if self.p_explore>0:
            explore_library = self.library[~self.library.index.isin(idxs)].sample(n=n_explore)
            new_library = pd.concat([new_library, explore_library])

        if self.log:
            old_library = self.library[~self.library.index.isin(new_library.index)]
            self.old_library = pd.concat([self.old_library,
                                          old_library[['smiles', 'score']]])
            self.old_library.drop_duplicates(subset='smiles', inplace=True)
            self.old_library.reset_index(inplace=True, drop=True)

        self.library = new_library

        self.library.reset_index(inplace=True, drop=True)
        gc.collect()
        end = time.time()
        self.timelog['prune_library'].append(start-end)

    def reset_library(self):
        if self.log:
            self.old_library = pd.concat([self.old_library,
                                          self.library[['smiles', 'score']]])
            self.old_library.drop_duplicates(subset='smiles', inplace=True)
            self.old_library.reset_index(inplace=True, drop=True)

        self.library = pd.DataFrame([], columns=['smiles', 'mols', 'score'])

    def add_data(self, smiles):
        smiles = to_smiles(smiles)
        smiles = self.clean_library(smiles)
        self.append_data(smiles)

    def append_data(self, smiles):
        start = time.time()
        mols = to_mols(smiles)
        bad_idxs = set([i for i in range(len(mols)) if mols[i] is None])
        smiles = [smiles[i] for i in range(len(smiles)) if not i in bad_idxs]
        mols = [mols[i] for i in range(len(mols)) if not i in bad_idxs]

        df = pd.DataFrame([[smiles[i], mols[i], None] for i in range(len(smiles))],
                          columns=['smiles', 'mols', 'score'])
        df = df[~df.smiles.isin(self.library.smiles)]
        self.library = pd.concat([self.library, df])
        self.library.reset_index(inplace=True, drop=True)
        self.score_library()
        end = time.time()
        self.timelog['append_data'].append(start-end)