# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_template.blocks.ipynb (unless otherwise specified).

__all__ = ['Block', 'ConstantBlock', 'ConstantMolBlock']

# Cell
from ..imports import *
from ..core import *
from ..chem import *
from .filters import *
from .template import *

# Cell

class Block():
    def __init__(self, template, links, name, subblocks=[]):
        self.template = template
        self.links = links
        self.name = name
        self.subblocks = subblocks
        self.sublinks = []

    def update_links(self):
        # updates self.sublinks based on self.subblocks
        raise NotImplementedError

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

    def eval_mol(self, mol):
        mol = to_mol(mol)
        smile = to_smile(mol)

        if self.match_fragment(smile):
            hardpass = self.template(mol, filter_type='hard')
        else:
            hardpass = False

        if hardpass:
            score = self.template(mol, filter_type='soft')
        else:
            score = self.template.failscore

        return [hardpass, score]

    def add_mapping(self, fragment):
        # converts unmapped fragment to mapped fragment
        raise NotImplementedError

    def remove_mapping(self, fragment):
        # converts mapped fragment to unmapped fragment
        raise NotImplementedError

    def shuffle_mapping(sef, fragment):
        # shuffles mapping of attachment points
        raise NotImplementedError

    def check_num_links(self, fragment):
        # checks if the fragment has the same number of attachments as defined in self.links
        raise NotImplementedError

    def process_fragment(self, fragment):
        # checks if fragment has correct number of attachments and maps fragment

        if self.check_num_links(fragment):
            fragment = self.add_mapping(fragment)
            attachment_pass = True
        else:
            attachment_pass = False

        return [attachment_pass, fragment]

    def load_data(self, fragments, recurse=False):
        # checks fragment attachments, then sends to template `load_data`
        # optionally recursive
        if recurse:
            for b in self.subblocks:
                b.load_data(fragments, recurse=True)

        fragments = maybe_parallel(self.process_fragment, fragments)
        fragments = [i[1] for i in fragments if i[0]]
        self.template.screen_mols(fragments)

    def sample(self, n, log='hard'):
        return self.template.sample(n, log=log)

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

    def recurse_fragments(self, fragment):
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

class ConstantBlock():
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

    def sample_data(self, n):
        return pd.DataFrame([self.constant, 0.]*n, columns=['smiles', 'final'])

    def __repr__(self):

        rep_str = f'Constant Block: {self.constant}'

        return rep_str


class ConstantMolBlock(ConstantBlock):
    def __init__(self, smile, name):
        super().__init__(smile, name)
        self.smile = canon_smile(smile)
        if '[*' in self.smile:
            self.smile = self.smile.replace('[*', '[0*')
        self.pattern = re.compile('\[(.*?)\*:(.*?)]')
        self.links = self.pattern.findall(smile)

    def sample_smile(self, n):
        return [self.smile]*n

    def __repr__(self):

        rep_str = f'Constant Block: {self.smile}'

        return rep_str