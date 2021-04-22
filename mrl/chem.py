# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_chem.ipynb (unless otherwise specified).

__all__ = ['to_mol', 'to_smile']

# Cell
from .core import *
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdMMPA
from rdkit.Chem.FilterCatalog import *

# Cell
def to_mol(smile_or_mol):
    if type(smile_or_mol) == str:
        mol = Chem.MolFromSmiles(smile_or_mol)
    else:
        mol = smile_or_mol

    return mol

def to_smile(smile_or_mol):
    if type(smile_or_mol) == str:
        smile = smile_or_mol
    else:
        smile = Chem.MolToSmiles(smile_or_mol)

    return smile