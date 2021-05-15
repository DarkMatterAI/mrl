# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/07_dataloaders.ipynb (unless otherwise specified).

__all__ = ['SMILES_CHAR_VOCAB', 'SPECIAL_TOKENS', 'MAPPING_TOKENS', 'HALOGEN_REPLACE', 'MAPPING_REPLACE', 'SMILE_REGEX',
           'MAPPING_REGEX', 'tokenize_by_character', 'tokenize_with_replacements', 'regex_tokenize', 'Vocab',
           'CharacterVocab', 'CharacterReplaceVocab', 'RegexVocab', 'test_reconstruction', 'batch_sequences',
           'lm_collate', 'sequence_prediction_collate', 'vector_collate', 'vector_reconstruction_collate',
           'vector_prediction_collate', 'BaseDataset', 'TextDataset', 'TextPredictionDataset', 'Vector_Dataset',
           'Vec_Recon_Dataset', 'Vec_Prediction_Dataset']

# Cell
from .imports import *
from .torch_imports import *
from .torch_core import *

# Cell

SMILES_CHAR_VOCAB = ['#', '(', ')', '+', '-', '/', '0',
                 '1', '2', '3', '4', '5', '6', '7',
                 '8', '=', '@', 'B', 'C', 'F', 'H',
                 'I', 'N', 'O', 'P', 'S', '[', '\\',
                 ']', 'c', 'i', 'l', 'n', 'o', 'r', 's',
                 '*', ':']


SPECIAL_TOKENS = ['bos', 'eos', 'pad', 'unk']

MAPPING_TOKENS = ['[1*:1]', '[2*:1]', '[1*:2]', '[2*:2]', '[1*:3]',
                  '[2*:3]', '[1*:4]', '[2*:4]', '[1*:5]', '[2*:5]']

HALOGEN_REPLACE = {'Br':'R',
                   'Cl':'L'}

MAPPING_REPLACE = {'[1*:1]':'A',
                   '[2*:1]':'D',
                   '[1*:2]':'E',
                   '[2*:2]':'G',
                   '[1*:3]':'J',
                   '[2*:3]':'M',
                   '[1*:4]':'Q',
                   '[2*:4]':'T',
                   '[1*:5]':'U',
                   '[2*:5]':'V'}


# Cell

SMILE_REGEX = """(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|H|\(|\)|\.|=|
                 #|-|\+|\\\\|\/|:|~|@|\?|>|#|\*|\$|\%[0-9]{2}|[0-9])"""

MAPPING_REGEX = """(\[.\*:.]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|H|\[|\]|\(|\)|\.|=|
                    #|-|\+|\\\\|\/|:|~|@|\?|>|#|\*|\$|\%[0-9]{2}|[0-9])"""

# Cell

def tokenize_by_character(input):
    "Splits `input` into inividual characters"
    return [i for i in input]

def tokenize_with_replacements(input, replacement_dict):
    "Replaces substrings in `input` using `replacement_dict`, then tokenizes by character"
    for k,v in replacement_dict.items():
        input = input.replace(k,v)
    return [i for i in input]

def regex_tokenize(input, regex):
    'Uses `regex` to tokenize `input`'
    tokens = [token for token in regex.findall(input)]
    return tokens

# Cell

class Vocab():
    '''
    Vocab - base vocabulary class

    Inputs:

        `itos` - list, list of tokens in vocabulary
    '''
    def __init__(self, itos):
        self.special_tokens = ['bos', 'eos', 'pad', 'unk']

        self.itos = self.special_tokens + [i for i in itos if not i in self.special_tokens]
        self.stoi = {self.itos[i]:i for i in range(len(self.itos))}
        self.unks = []

    def tokenize(self, input):
        'Tokenize `input`'
        raise NotImplementedError

    def numericalize(self, input):
        'Numericalize `input` into integers'
        output = []
        for tok in input:
            if tok in self.stoi.keys():
                output.append(self.stoi[tok])
            else:
                output.append(self.stoi['unk'])
                self.unks.append(tok)
        return output

    def reconstruct(self, input):
        'Reconstruct `input` into a string'
        output = []
        for item in input:
            item = self.itos[item]
            if item=='eos':
                break

            if not item=='bos':
                output.append(item)

        return ''.join(output)

    def update_vocab(self):
        'Adds tokens in `self.unks` to vocabulary'
        unks = list(set(self.unks))
        self.itos += unks
        self.stoi = {self.itos[i]:i for i in range(len(self.itos))}
        self.unks = []

    def update_vocab_from_data(self, inputs):
        'Tokenizes `inputs` and updates the vocabulary with any unknown tokens'
        _ = [self.numericalize(self.tokenize(i)) for i in inputs]
        self.update_vocab()


class CharacterVocab(Vocab):
    '''
    CharacterVocab - tokenize by character
    '''
    def tokenize(self, input):
        toks = tokenize_by_character(input)
        toks = ['bos'] + toks + ['eos']
        return toks


class CharacterReplaceVocab(Vocab):
    '''
    CharacterReplaceVocab - tokenize by character with replacement

    Inputs:

        `itos` - list, list of tokens
        `replace_dict` - dict, replacement dictionary of the form {multi_character_token : single_character_token}.
        ie replace_dict={'Br':'R', 'Cl':'L'}
    '''
    def __init__(self, itos, replace_dict):
        itos = list(itos)
        self.replace_dict = replace_dict
        self.reverse_dict = {v:k for k,v in replace_dict.items()}
        for rep in self.reverse_dict.keys():
            if not rep in itos:
                itos.append(rep)
        super().__init__(itos)

    def tokenize(self, smile):
        toks = tokenize_with_replacements(smile, self.replace_dict)
        toks = ['bos'] + toks + ['eos']
        return toks

    def reconstruct(self, input):
        output = []
        for item in input:
            item = self.itos[item]
            if item=='eos':
                break

            if not item=='bos':
                if item in self.reverse_dict.keys():
                    item = self.reverse_dict[item]

                output.append(item)

        return ''.join(output)


class RegexVocab(Vocab):
    '''
    RegexVocab - tokenize using `pattern`

    Inputs:

        `itos` - list, list of tokens
        `pattern` - str, regex string
    '''
    def __init__(self, itos, pattern):
        super().__init__(itos)

        self.pattern = pattern
        self.regex = re.compile(self.pattern)

    def tokenize(self, smile):
        toks = regex_tokenize(smile, self.regex)
        toks = ['bos'] + toks + ['eos']
        return toks

# Cell

def test_reconstruction(vocab, inputs):
    "Returns all items in `inputs` that can't be correctly reconstructed using `vocab`"
    fails = []
    for item in inputs:
        recon = vocab.reconstruct(vocab.numericalize(vocab.tokenize(item)))
        if not item==recon:
            fails.append((item, recon))

    return fails

# Cell

def batch_sequences(sequences, pad_idx):
    'Packs `sequences` into a dense tensor, using `pad_idx` for padding'
    max_len = max([len(i) for i in sequences])+1
    bs = len(sequences)

    batch_tensor = torch.zeros((bs, max_len)).long() + pad_idx

    for i,item in enumerate(sequences):
        batch_tensor[i,:item.shape[0]] = item

    return batch_tensor


def lm_collate(batch, pad_idx, batch_first=True):
    '''
    Collate function for language models. Returns packed
    batch for next-token prediction
    '''
    batch_tensor = batch_sequences(batch, pad_idx)

    if batch_first:
        output = (batch_tensor[:,:-1], batch_tensor[:,1:])
    else:
        batch_tensor = batch_tensor.T
        output = (batch_tensor[:-1,:], batch_tensor[1:,:])

    return to_device(output)

def sequence_prediction_collate(batch, pad_idx, batch_first=True):
    '''
    Collate function for predicting some y value from a sequence
    '''
    batch_tensor = batch_sequences([i[0] for i in batch], pad_idx)
    y_vals = torch.stack([i[1] for i in batch])
    y_vals = y_vals.squeeze(-1)

    if not batch_first:
        batch_tensor = batch_tensor.T

    return to_device((batch_tensor, y_vals))

def vector_collate(batch):
    '''
    Collate function for vectors
    '''
    fps = torch.stack(batch)
    return to_device(fps)

def vector_reconstruction_collate(batch, pad_idx, batch_first=True):
    '''
    Collate function for predicting a sequence from an input vector where
    `batch_tensor` is needed for input (ie predict SMILES from properties)
    '''
    fps = torch.stack([i[0] for i in batch])
    batch_tensor = batch_sequences([i[1] for i in batch], pad_idx)

    if batch_first:
        output = ((batch_tensor[:,:-1], fps), batch_tensor[:,1:])
    else:
        batch_tensor = batch_tensor.T
        output = ((batch_tensor[:-1,:], fps), batch_tensor[1:,:])

    return to_device(output)

def vector_prediction_collate(batch):
    '''
    Collate function for predicting some y value from a vector
    '''
    fps = torch.stack([i[0] for i in batch])
    y_vals = torch.stack([i[1] for i in batch])
    y_vals = y_vals.squeeze(-1)
    return to_device((fps, y_vals))


# Cell

class BaseDataset(Dataset):
    '''
    BaseDataset - base dataset

    Inputs:

        `collate_function` - batch collate function for the particular dataset class
    '''
    def __init__(self, collate_function):
        self.collate_function = collate_function

    def dataloader(self, bs, num_workers=-1, **dl_kwargs):
        if num_workers==-1:
            if 'ncpus' in os.environ.keys():
                num_workers = int(os.environ['ncpus'])
            else:
                num_workers=os.cpu_count()

        return DataLoader(self, batch_size=bs, num_workers=num_workers,
                          collate_fn=self.collate_function, **dl_kwargs)

    def new(self):
        raise NotImplementedError

# Cell

class TextDataset(BaseDataset):
    '''
    TextDataset - base dataset for language modes

    Inputs:

        `smiles` - list[str], list of text sequences

        `vocab` - Vocab, vocabuary for tokenization/numericaization

        `collate_function` - batch collate function. If None, defauts to `lm_collate`
    '''
    def __init__(self, smiles, vocab, collate_function=None):
        self.smiles = smiles
        self.vocab = vocab
        if collate_function is None:
            collate_function = partial(lm_collate, pad_idx=self.vocab.stoi['pad'])

        super().__init__(collate_function)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        tokens = self.vocab.tokenize(smile)
        ints = self.vocab.numericalize(tokens)
        ints = torch.LongTensor(ints)
        return ints

    def new(self, smiles):
        return self.__class__(smiles, self.vocab, self.collate_function)

# Cell

class TextPredictionDataset(TextDataset):
    '''
    TextDataset - base dataset for predicting from text strings

    Inputs:

        `smiles` - list[str], list of text sequences

        `y_vals` - list[int, float], list of paired output values

        `vocab` - Vocab, vocabuary for tokenization/numericaization

        `collate_function` - batch collate function. If None, defauts to `sequence_prediction_collate`
    '''
    def __init__(self, smiles, y_vals, vocab, collate_function=None):

        if collate_function is None:
            collate_function = partial(sequence_prediction_collate, pad_idx=vocab.stoi['pad'])

        super().__init__(smiles, vocab, collate_function)

        self.y_vals = y_vals

    def __getitem__(self, idx):
        ints = super().__getitem__(idx)
        y_val = torch.Tensor([self.y_vals[idx]]).float()
        return (ints, y_val)

    def new(self, smiles, y_vals):
        return self.__class__(smiles, y_vals, self.vocab, self.collate_function)

# Cell

class Vector_Dataset(BaseDataset):
    '''
    Vector_Dataset - base dataset for molecule-derived vectors

    Inputs:

        `smiles` - list[str], list of text sequences

        `mol_function` - function to convert smiles to a vector

        `collate_function` - batch collate function. If None, defauts to `vector_collate`
    '''
    def __init__(self, smiles, mol_function, collate_function=None):
        if collate_function is None:
            collate_function = vector_collate
        super().__init__(collate_function)

        self.smiles = smiles
        self.mol_function = mol_function

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        vec = self.mol_function(smile)
        vec = torch.FloatTensor(vec)
        return vec

    def new(self, smiles):
        return self.__class__(smiles, self.mol_function, self.collate_function)

# Cell

class Vec_Recon_Dataset(Vector_Dataset):
    '''
    Vec_Recon_Dataset - base dataset for predicting smiles from molecule-derived vectors

    Inputs:

        `smiles` - list[str], list of text sequences

        `vocab` - Vocab, vocabuary for tokenization/numericaization

        `mol_function` - function to convert smiles to fingerprints

        `collate_function` - batch collate function. If None, defauts to `vector_reconstruction_collate`
    '''
    def __init__(self, smiles, vocab, mol_function, collate_function=None):

        if collate_function is None:
            collate_function = partial(vector_reconstruction_collate, pad_idx=vocab.stoi['pad'])

        super().__init__(smiles, mol_function, collate_function)
        self.vocab = vocab

    def __getitem__(self, idx):
        smile = self.smiles[idx]

        vec = self.mol_function(smile)
        vec = torch.FloatTensor(vec)

        tokens = self.vocab.tokenize(smile)
        ints = self.vocab.numericalize(tokens)
        ints = torch.LongTensor(ints)

        return (vec, ints)

    def new(self, smiles):
        return self.__class__(smiles, self.vocab, self.mol_function, self.collate_function)

# Cell

class Vec_Prediction_Dataset(Vector_Dataset):
    '''
    Vec_Prediction_Dataset - base dataset for predicting y_vals from molecule derived vectors

    Inputs:

        `smiles` - list[str], list of text sequences

        `y_vals` - list[int, float], list of paired output values

        `mol_function` - function to convert smiles to fingerprints

        `collate_function` - batch collate function. If None, defauts to `vector_prediction_collate`
    '''
    def __init__(self, smiles, y_vals, mol_function, collate_function=None):
        if collate_function is None:
            collate_function = vector_prediction_collate
        super().__init__(smiles, mol_function, collate_function)

        self.y_vals = y_vals

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        fp = super().__getitem__(idx)
        y_val = torch.FloatTensor([self.y_vals[idx]])
        return (fp, y_val)

    def new(self, smiles, y_vals):
        return self.__class__(smiles, y_vals, self.mol_function, self.collate_function)