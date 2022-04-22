# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_template.blocks.ipynb (unless otherwise specified).

__all__ = ['Block', 'MolBlock', 'ConstantBlock', 'ConstantMolBlock', 'BlockTemplate', 'RGroupBlockTemplate',
           'DoubleRGroupBlockTemplate', 'LinkerBlockTemplate', 'ScaffoldBlockTemplate']

# Cell
from ..imports import *
from ..core import *
from ..chem import *
from .filters import *
from .template import *

# Cell

class Block():
    '''
    Block - base class for Blocks

    Inputs:

    - `template Template`: `Template` subclass

    - `links list[str]`: list, defines links between this block and other blocks

    - `name str`: block name

    - `subblocks list[Block]`: list of `Block` classes nested within this block
    '''
    def __init__(self, template, links, name, subblocks=None):
        if subblocks is None:
            subblocks = []

        self.template = template
        self.links = links
        self.name = name
        self.subblocks = subblocks
        self.sublinks = []
        self.update_links()

    def update_links(self):
        # grabs all subblock links
        for b in self.subblocks:
            self.sublinks.append(b.links)
            for sl in b.sublinks:
                self.sublinks.append(sl)

    def eval_mol(self, mol, previous_pass=True):
        '''
        eval_mol - evaluates `mol`.

        If `mol` passes the hard filters in `self.template`, it is scored by `self.template.soft_filters`.
        If not, `self.template.failscore` is given instead.

        Returns `hardpass` (result of the hard filters), `score`, and logging information

        Context: this will be executed in parallel processing if available, meaning the automated
        logging implemented in `Template.__call__` won't work. For this reason, the log information
        is captured as an output and added to the template log later (see `BlockTree`)
        '''
        mol = self.template.to_mol(mol)
        smile = self.template.to_string(mol)

        if type(smile)==str:
            match = self.match_fragment(smile)
        else:
            match = False

        if previous_pass and match:
            hardpass, hardlog = self.template.hf(mol)
        else:
            hardpass = False
            hardlog = []

        if hardpass:
            score, softlog = self.template.sf(mol)
        else:
            score = self.template.fail_score
            softlog = []

        return [hardpass, score, hardlog, softlog]

    def match_fragment(self, fragment):
        # determine if fragment matches block link pattern
        raise NotImplementedError

    def match_fragment_recursive(self, fragment):
        # recursively match fragment to all subblocks
        if self.match_fragment(fragment):
            output = True
        else:
            output = False
            for block in self.subblocks:
                if block.match_fragment_recursive(fragment):
                    output = True

        return output

    def sample(self, n, log='hard'):
        # wrapper for template log sampling
        return self.template.sample(n, log=log)

    def load_data(self, fragments, recurse=False):
        # checks fragment attachments, then sends to template `load_data`
        # optionally recursive
        if recurse:
            for b in self.subblocks:
                b.load_data(fragments, recurse=True)

        matches = maybe_parallel(self.match_fragment, fragments)
        fragments = [fragments[i] for i in range(len(fragments)) if matches[i]]
        self.template.screen_mols(fragments)

    def decompose_fragments(self, fragment_string):
        # decomposes a string of multiple fragments into a list of single fragments
        raise NotImplementedError

    def join_fragments(self, fragment_list):
        # joins list of fragments into single string
        raise NotImplementedError

    def fuse_fragments(self, fragment_string):
        # fuses fragment string into single output
        raise NotImplementedError

    def join_and_fuse(self, fragment_list):
        return self.fuse_fragments(self.join_fragments(fragment_list))

    def recurse_fragments(self, fragment, add_constant=True):
        # recursively break down fragments, route to subblocks, fuse and evaluate
        raise NotImplementedError

    def __repr__(self):

        rep_str = f'Block {self.name}: {self.links}\n\t' + '\n\t'.join(self.template.__repr__().split('\n'))

        if self.subblocks:
            rep_str += '\n'
            for b in self.subblocks:
                rep_str += '\n\t' + '\n\t'.join(b.__repr__().split('\n'))

        return rep_str


# Cell

class MolBlock(Block):
    '''
    MolBlock - `Block` subclass specific to working with smiles strings. This class expects
    links between SMILES fragments to be wildcard atoms of the form `{isotope}*:{map_number}`.
    Note that '0' should not be used as an isotope number because RDKit removes '0' isotopes from
    SMILES strings automatically.
    '''
    def __init__(self, template, links, name, subblocks=None):
        super().__init__(template, links, name, subblocks=subblocks)

        # self.links = ['1*:2', '1*:3']
        self.pattern = re.compile('\[.\*:.]')

        for link in self.links:
            assert not '0*' in link, "Do not use 0 as an isotope, RDKit automatically removes it"

    def pattern_match(self, fragment):
        "extracts `{isotope}*:{map_number}` substrings from `fragment`"
        matches = self.pattern.findall(fragment)
        return [i[1:-1] for i in matches]

    def is_mapped(self, fragment):
        'Validates all wildcards are mapped with the form `{isotope}*:{map_number}`'
        if fragment.count('*') == len(self.pattern_match(fragment)):
            mapped = True
        else:
            mapped = False

        return mapped

    def add_mapping(self, fragment, links=None):
        "Maps wildcards in `fragment` (ie changes `*` to `{isotope}*:{map_number}`)"
        if self.is_mapped(fragment):
            # already mapped
            mapped = fragment
        else:
            if len(self.pattern_match(fragment))>0:
                # partially mapped, something went wrong
                fragment = self.remove_mapping(fragment)

            if links is None:
                links = list(self.links)
                random.shuffle(links)

            mapped = ''
            link_count = 0
            for s in fragment:
                if s=='*':
                    s = f'[{links[link_count]}]'
                    link_count += 1
                mapped += s

        return mapped

    def remove_mapping(self, fragment):
        "Converts mappings from `{isotope}*:{map_number}` to `*`"
        matches = self.pattern_match(fragment)
        for match in matches:
            fragment = fragment.replace(f'[{match}]', '*')
        return fragment

    def match_fragment(self, fragment):
        "Determines if `fragment` matches the specification in `self.links`"
        match = False
        if fragment.count('*') == len(self.links):
            if not self.is_mapped(fragment):
                fragment = self.add_mapping(fragment)

            matches = self.pattern_match(fragment)
            if len(matches)==len(set(matches)) and set(matches)==set(self.links):
                match = True

        return match

    def _load_fragment(self, fragment):
        if fragment.count('*') == len(self.links):
            fragment = self.add_mapping(fragment)
            fragpass = True
        else:
            fragpass = False

        return [fragment, fragpass]

    def load_data(self, fragments, recurse=False):
        '''
        load_data - checks if elements in `fragments` match `self.links`, then
        passes matching fragments to `self.template` for screening and scoring
        '''
        if recurse:
            for b in self.subblocks:
                b.load_data(fragments, recurse=True)

        fragments = maybe_parallel(self._load_fragment, fragments)
        fragments = [i[0] for i in fragments if i[1]]
        self.template.screen_mols(fragments)

    def sample_smiles(self, n, log='hard'):
        return self.template.sample_smiles(n, log=log)

    def shuffle_mapping(self, fragment):
        'Shuffles map numbers on `fragment`'
        current_mapping = self.pattern_match(fragment)
        new_mapping = list(current_mapping)
        random.shuffle(new_mapping)

        fragment = self.remove_mapping(fragment)
        fragment = self.add_mapping(fragment, links=new_mapping)
        return fragment

    def decompose_fragments(self, fragment_string):
        return fragment_string.split('.')

    def join_fragments(self, fragment_list):
        return '.'.join(fragment_list)

    def fuse_fragments(self, fragment_string):
        try:
            new_smile = fuse_on_atom_mapping(fragment_string)
        except:
            new_smile = ''
        return new_smile

    def recurse_fragments(self, fragments, add_constant=True):
        '''
        recurse_fragments - recursively evlauates `fragments` against `self.template`
        and all blocks in `self.subblocks`

        Inputs:

        - `fragments [str, list[str]]`: fragments to process.
        Can either be a single string of the form `'f1.f2.f3'`
        or a list of the form `['f1','f2','f3']`. All items in
        `fragments` should correspond to the same final molecule.

        - `add_constant bool`: If True, constant sequences in any `ConstantBlock` subclasses
        are added to `fragments` during evaluation. Should be `True` if constant sequences are
        missing from `fragments` or False if they are present

        Returns:

        - `fused str`: fragments fused at this stage

        - `total_pass bool`: True if `fragments` passed all subblock
        templates and `fused` passed `self.template`

        - `total_score float`: sum of scores from `self.template.soft_filters` and subblock template soft filters

        - `output_dicts list[dict]`: list of dictionaries holding information from this block and subblocks

        Recurse fragments works in the following way:

            1. Fragments are decomposed based on `self.decompose_fragments`
            2. Fragments are routed to subblocks if present using `self.match_fragment_recursive`
            3. Any fragments matching a subblock are first evaluated by that subblock's template
            4. If `add_constant=True`, constant sequences from any `ConstantBlock` subblocks are added
            5. Fragments are joined and fused using `self.join_fragments` and `self.fuse_fragments`
            6. The fused fragments are processed by `self.eval_mol`
        '''
        output_dicts = []
        total_pass = []
        total_score = 0.

        if not is_container(fragments):
            fragments = [fragments]

        valids = self.template.validate(fragments, cpus=0)

        if all(valids):

            fragments = [self.decompose_fragments(i) for i in fragments]
            fragments = [item for sublist in fragments for item in sublist]

            if self.subblocks:
                new_fragments = []

                unrouted = list(fragments) # copy list

                for sb in self.subblocks:
                    routed = [i for i in unrouted if sb.match_fragment_recursive(i)]
                    unrouted = [i for i in unrouted if not i in routed]

                    if routed:
                        r_fused, r_pass, r_score, subdicts = sb.recurse_fragments(routed)
                        new_fragments.append(r_fused)
                        total_pass.append(r_pass)
                        total_score += r_score
                        output_dicts += subdicts

                    if isinstance(sb, ConstantBlock) and add_constant:
                        new_fragments.append(sb.smile)

                fragments = new_fragments + unrouted

            joined_fragments = self.join_fragments(fragments)
            fused = self.fuse_fragments(joined_fragments)

            frag_pass, frag_score, hardlog, softlog = self.eval_mol(fused, all(total_pass))
            total_pass.append(frag_pass)
            total_score += frag_score

            total_pass = all(total_pass)

            output_dict = {
                'block' : self.name,
                'fused' : fused,
                'fragments' : fragments,
                'block_pass' : frag_pass,
                'block_score' : frag_score,
                'all_pass' : total_pass,
                'all_score' : total_score,
                'hardlog' : hardlog,
                'softlog' : softlog
            }

            output_dicts.append(output_dict)
        else:
            fused = ''
            total_pass = False
            total_score = self.template.fail_score
            output_dicts = {}

        return fused, total_pass, total_score, output_dicts


# Cell

class ConstantBlock():
    '''
    ConstantBlock - base block class for constant sequence
    '''
    def __init__(self, constant, name):
        self.constant = constant
        self.name = name
        self.links = []
        self.subblocks = []
        self.sublinks = []

    def match_fragment(self, fragment):
        return False

    def match_fragment_recursive(self, fragment):
        return False

    def load_data(self, fragments, recurse=False):
        pass

    def sample(self, n):
        return pd.DataFrame([[self.constant, 0.] for i in range(n)], columns=['smiles', 'final'])

    def __repr__(self):

        rep_str = f'Constant Block: {self.constant}'

        return rep_str


class ConstantMolBlock(ConstantBlock):
    '''
    ConstantMolBlock - constant block for SMILES sequence
    '''
    def __init__(self, smile, name, links=None):
        super().__init__(smile, name)
        self.smile = canon_smile(smile)
        if '[*' in self.smile:
            self.smile = self.smile.replace('[*', '[0*')
        self.pattern = re.compile('\[.\*:.]')

        if links is None:
            links = [i[1:-1] for i in self.pattern.findall(smile)]

        self.links = links

    def sample_smiles(self, n):
        return [self.smile]*n

    def __repr__(self):

        rep_str = f'Constant Block: {self.smile}'

        return rep_str


# Cell

class BlockTemplate():
    '''
    BlockTemplate - base class for handling nested blocks. Takes care of running fragment strings and logging outputs
    '''
    def __init__(self, head_block, lookup=True):
        self.head_block = head_block
        self.nodes = self.nodes_to_list(self.head_block)
        self.leaf_nodes = [i for i in self.nodes if not i.subblocks]
        self.live_leafs = [i for i in self.leaf_nodes if not isinstance(i, ConstantBlock)]
        self.node_dict = {i.name:i for i in self.nodes}
        self.log = []
        self.block_log = []
        self.input_id = 0
        self.lookup = lookup
        self.lookup_table = {}

    def __call__(self, fragments, filter_type='hard', add_constant=True):

        outputs = self.recurse_fragments(fragments, add_constant=add_constant)

        if filter_type=='hard':
            outputs = [i[2] for i in outputs]
        else:
            outputs = [i[3] for i in outputs]

        return outputs

    def standardize(self, smiles):
        return self.head_block.template.standardize(smiles)

    def validate(self, smiles):
        return self.head_block.template.validate(smiles)

    def eval_mols(self, mols):
        return self.__call__(mols, filter_type='soft')

    def nodes_to_list(self, block):
        nodes = [block]
        if block.subblocks:
            for subblock in block.subblocks:
                nodes += self.nodes_to_list(subblock)

        return nodes

    def log_outputs(self, outputs):
        'routes log data from `recurse_fragments` to relevant templates'
        log_dict = {}

        for output_dict in outputs:
            if not output_dict['block'] in log_dict.keys():
                log_dict[output_dict['block']] = {'hard':[],
                                                  'soft':[]}

            if output_dict['hardlog']:
                log_dict[output_dict['block']]['hard'].append(output_dict['hardlog'])

            if output_dict['softlog']:
                log_dict[output_dict['block']]['soft'].append(output_dict['softlog'])

            output_dict.pop('hardlog')
            output_dict.pop('softlog')

            self.block_log.append(output_dict)

        for blockname in log_dict.keys():
            node = self.node_dict[blockname]
            if not log_dict[blockname]['hard']==[]:
                node.template.log_data(log_dict[blockname]['hard'], filter_type='hard')

            if not log_dict[blockname]['soft']==[]:
                node.template.log_data(log_dict[blockname]['soft'], filter_type='soft')

    def recurse_fragments(self, fragments, add_constant=True):
        '''
        Recurses fragments through the block tree, then logs results.

        See `MolBlock.recurse_fragments`
        '''

        if not is_container(fragments):
            fragments = [fragments]

        output_data = [[] for i in fragments]

        if self.lookup:
            to_screen = []
            idx_list = []
            for i, frag in enumerate(fragments):
                if frag in self.lookup_table.keys():
                    output_data[i] = self.lookup_table[frag]
                else:
                    to_screen.append(frag)
                    idx_list.append(i)
        else:
            to_screen = fragments

        outputs = maybe_parallel(self.head_block.recurse_fragments, to_screen, add_constant=add_constant)
        output_dicts = []

        for i, out in enumerate(outputs):
            fused, allpass, allscore, log_dicts = out

            output_data[idx_list[i]] = [to_screen[i], fused, allpass, allscore]

            if self.lookup:
                self.lookup_table[to_screen[i]] = [to_screen[i], fused, allpass, allscore]

#             output_data.append([fragments[i], fused, allpass, allscore])

            for ld in log_dicts:
                ld['input_id'] = self.input_id

            self.input_id += 1
            output_dicts += log_dicts

        self.log_outputs(output_dicts)
        self.log += output_data

        return output_data

    def load_data(self, fragments, recurse=False):
        self.head_block.load_data(fragments, recurse=recurse)

    def _sample_leaf_nodes(self, include_constant=False):
        if include_constant:
            leaf_nodes = self.leaf_nodes
        else:
            leaf_nodes = self.live_leafs

        output = []

        for node in leaf_nodes:
            output.append(node.sample(1).values[:,0])

        output = flatten_list_of_lists([list(i) for i in output])

        return output

    def sample_leaf_nodes(self, n, include_constant=False, join=True):
        'Sample leaf nodes'
        samples = maybe_parallel(self._sample_leaf_nodes, [include_constant]*n)
        if join:
            samples = [self.head_block.join_fragments(i) for i in samples]
        return samples

    def save(self, filename, with_data=True):
        '''
        save - save `BlockTemplate` object

        Inputs

        - 'filename str': save filename

        - `with_data bool`: if True BlockTemplate is saved with logged data
        '''

        if not with_data:
            log = self.log
            self.log = []

            block_log = self.block_log
            self.block_log = []

            data_dict = {}
            for nodename, node in self.node_dict.items():
                if hasattr(node, 'template'):
                    data_dict[nodename] = {'hard':node.template.hard_log,
                                           'soft':node.template.soft_log}
                    node.template.clear_data()

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        if not with_data:
            self.log = log
            self.block_log = block_log

            for nodename, node in self.node_dict.items():
                if hasattr(node, 'template'):
                    node.template.hard_log = data_dict[nodename]['hard']
                    node.template.soft_log = data_dict[nodename]['soft']

    @classmethod
    def from_file(cls, filename):
        'load template from file'
        template = pickle.load(open(filename, 'rb'))
        return template


    def __repr__(self):

        rep_str = 'Block Template' + '\n\t' + '\n\t'.join(self.head_block.__repr__().split('\n'))

        return rep_str


# Cell

class RGroupBlockTemplate(BlockTemplate):
    '''
    RGroupBlockTemplate - block template for r-group screening

    Inputs:

    - `base_smile str`: base smile to attach r-group to.
    Should have a single unmapped wildcard atom, ie `*CCCC`

    - `rgroup_template Template`: template for screening r-groups

    - `full_molecule_template Optional[Template]`: Optional template for full molecule
    '''
    def __init__(self, base_smile, rgroup_template, full_molecule_template=None,
                replace_wildcard=True, lookup=True):

        assert base_smile.count('*')==1, '`base_smile` should have exactly one wildcard'

        self.replace_wildcard = replace_wildcard

        mol = rgroup_template.to_mol(base_smile)

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)

        base_smile = rgroup_template.to_string(mol)
        base_smile = base_smile.replace('*', '[1*:1]')

        scaffold_block = ConstantMolBlock(base_smile, 'scaffold')
        rgroup_block = MolBlock(rgroup_template, ['2*:1'], 'rgroup')
        self.rgroup_block = rgroup_block

        if full_molecule_template is None:
            full_molecule_template = Template([], log=rgroup_template.log,
                                             use_lookup=rgroup_template.use_lookup,
                                             cpus=rgroup_template.cpus,
                                             mode=rgroup_template.mode)

        head_block = MolBlock(full_molecule_template, [], 'full_molecule', subblocks=[scaffold_block, rgroup_block])

        super().__init__(head_block, lookup)

    def recurse_fragments(self, fragments, add_constant=True):

        if self.replace_wildcard:
            new_fragments = []
            valids = self.validate(fragments)
            for i in range(len(fragments)):
                if valids[i] and self.rgroup_block.match_fragment(fragments[i]):
                    new_fragment = self.rgroup_block.add_mapping(fragments[i])
                else:
                    new_fragment = fragments[i]

                new_fragments.append(new_fragment)

            fragments = new_fragments

        return super().recurse_fragments(fragments, add_constant=add_constant)


# Cell

class DoubleRGroupBlockTemplate(BlockTemplate):
    '''
    DoubleRGroupBlockTemplate - block template for screening two r-groups

    Inputs:

    - `base_smile str`: base smile to attach r-group to.
    Should have two mapped wildcard atoms, ie `'c1nc2c([1*:1])cncc2cc1[1*:2]'`.
    Rgroup1 will be fused to wildcard `1*:1` and Rgroup 2
    will be fused to wildcard `1*:2`

    - `r1_template Template`: template for screening rgroup 1

    - `r2_template Template`: template for screening rgroup 2

    - `full_molecule_template Optional[Template]`: Optional template for full molecule
    '''
    def __init__(self, base_smile, r1_template, r2_template,
                 full_molecule_template=None, lookup=True):

        assert base_smile.count('*')==2, '`base_smile` should have exactly two wildcards'

        pattern = re.compile('\[.\*:.]')
        mapping = pattern.findall(base_smile)
        scaffold_links = ['[1*:1]', '[1*:2]']
        assert set(mapping)==set(scaffold_links), "`base_smile` must be mapped with ['1*:1', '1*:2']"

        base_smile = r1_template.standardize(base_smile)

        scaffold_block = ConstantMolBlock(base_smile, 'scaffold')
        r1_block = MolBlock(r1_template, ['2*:1'], 'r1')
        r2_block = MolBlock(r2_template, ['2*:2'], 'r2')

        if full_molecule_template is None:
            full_molecule_template = Template([], log=rgroup_template.log,
                                             use_lookup=rgroup_template.use_lookup,
                                             cpus=rgroup_template.cpus,
                                             mode=rgroup_template.mode)

        head_block = MolBlock(full_molecule_template, [], 'full_molecule',
                              subblocks=[scaffold_block, r1_block, r2_block])

        super().__init__(head_block, lookup)

# Cell

class LinkerBlockTemplate(BlockTemplate):
    '''
    LinkerBlockTemplate - block template for screening linkers

    Inputs:

    - `smile1 str`: left-side linker attachment.
    Should have a single unmapped wildcard, ie `*CCCC`

    - `smile2 str`: right-side linker attachment.
    Should have a single unmapped wildcard, ie `*CCCC`

    - `linker_template Template`: template for screening the linker

    - `full_molecule_template Optional[Template]`: Optional template for full molecule
    '''
    def __init__(self, smile1, smile2, linker_template,
                 full_molecule_template=None, lookup=True):
        assert smile1.count('*')==1, '`smile1` should contain 1 wildcard'
        assert smile2.count('*')==1, '`smile2` should contain 1 wildcard'

        mol1 = linker_template.to_mol(smile1)
        for atom in mol1.GetAtoms():
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)

        smile1 = linker_template.to_string(mol1)
        smile1 = smile1.replace('*', '[1*:1]')

        mol2 = linker_template.to_mol(smile2)
        for atom in mol2.GetAtoms():
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)

        smile2 = linker_template.to_string(mol2)
        smile2 = smile2.replace('*', '[1*:2]')

        self.smile1 = smile1
        self.smile2 = smile2

        block1 = ConstantMolBlock(smile1, 'left_linker')
        block2 = ConstantMolBlock(smile2, 'right_linker')
        linker_block = MolBlock(linker_template, ['2*:1', '2*:2'], 'linker')

        if full_molecule_template is None:
            full_molecule_template = Template([], log=rgroup_template.log,
                                             use_lookup=rgroup_template.use_lookup,
                                             cpus=rgroup_template.cpus,
                                             mode=rgroup_template.mode)

        head_block = MolBlock(full_molecule_template, [], 'full_molecule',
                              subblocks=[block1, block2, linker_block])

        super().__init__(head_block, lookup)

# Cell

class ScaffoldBlockTemplate(BlockTemplate):
    '''
    ScaffoldBlockTemplate - block template for screening scaffolds or rings with multiple attachments

    Inputs:

    - `attachments list`: list of mapped attachments.
    All attachments should have a single wildcard
    atom mapped following the format `[{isotope}*:{map_num}]`,
    ie `['[1:*1]CC', '[1:*2]CCC']`

    - `scaffold_template Template`: template for screening the scaffold

    - `full_molecule_template Optional[Template]`: Optional template for full molecule
    '''

    def __init__(self, attachments, scaffold_template,
                 full_molecule_template=None, lookup=True):

        pattern = re.compile('\[.\*:.]')
        links = []
        for att in attachments:
            matches = [i[1:-1] for i in pattern.findall(att)]
            assert len(matches)==1, (f'Attachment {att} should have one mapped '
                                    'wildcard of the form `[isotope*:map_number]`')
            links += matches

        scaffold_links = []

        for link in links:
            isotope, map_num = link.split('*:')
            if isotope==1:
                new_isotope=2
            else:
                new_isotope=1

            scaffold_links.append(f'{new_isotope}*:{map_num}')

        subblocks = [ConstantMolBlock(attachments[i], f'attachment_{i}') for i in range(len(attachments))]
        scaffold_block = MolBlock(scaffold_template, scaffold_links, 'scaffold')
        subblocks.append(scaffold_block)

        if full_molecule_template is None:
            full_molecule_template = Template([], log=rgroup_template.log,
                                             use_lookup=rgroup_template.use_lookup,
                                             cpus=rgroup_template.cpus,
                                             mode=rgroup_template.mode)

        head_block = MolBlock(full_molecule_template, [], 'full_molecule', subblocks=subblocks)

        super().__init__(head_block, lookup)