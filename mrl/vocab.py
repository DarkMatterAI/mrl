# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/07_vocab.ipynb (unless otherwise specified).

__all__ = ['SMILES_CHAR_VOCAB', 'SPECIAL_TOKENS', 'MAPPING_TOKENS', 'HALOGEN_REPLACE', 'MAPPING_REPLACE',
           'AMINO_ACID_VOCAB', 'NUCLEIC_ACID_VOCAB', 'NUCLEIC_ACID_EXPANDED_VOCAB', 'NUCLEIC_ACID_DIMERS',
           'NUCLEIC_ACID_TRIMERS', 'NUCLEIC_ACID_TRIMERS', 'DNA_VOCAB', 'DNA_DIMERS', 'DNA_TRIMERS', 'SELFIES_VOCAB',
           'SELFIES_VOCAB_LEGACY', 'SELFIES_EXPANDED_VOCAB_LEGACY', 'SELFIES_EXPANDED_VOCAB_LEGACY', 'pad_vocab',
           'SMILE_REGEX', 'MAPPING_REGEX', 'AA_MAPPING_REGEX', 'tokenize_by_character', 'tokenize_with_replacements',
           'regex_tokenize', 'tokenize_by_kmer', 'Vocab', 'CharacterVocab', 'FuncVocab', 'SelfiesVocab',
           'CharacterReplaceVocab', 'RegexVocab', 'KmerVocab', 'test_reconstruction']

# Cell
from .imports import *

# Cell

SMILES_CHAR_VOCAB = ['#', '(', ')', '+', '-', '/', '0',
                 '1', '2', '3', '4', '5', '6', '7',
                 '8', '9', '=', '@', 'B', 'C', 'F', 'H',
                 'I', 'N', 'O', 'P', 'S', '[', '\\',
                 ']', 'c', 'i', 'l', 'n', 'o', 'r', 's',
                 '*', ':', '.', 'a', 'K', 'e']


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

AMINO_ACID_VOCAB = ['A', 'C', 'D', 'E', 'F',
                     'G', 'H', 'I', 'K', 'L',
                     'M', 'N', 'P', 'Q', 'R',
                     'S', 'T', 'V', 'W', 'Y']

NUCLEIC_ACID_VOCAB = ['A', 'C', 'G', 'T', 'U', 'N']

NUCLEIC_ACID_EXPANDED_VOCAB = ['A', 'C', 'G', 'T',
                              'U', 'M', 'R', 'W',
                              'S', 'Y', 'K', 'V',
                              'H', 'D', 'B', 'N']

NUCLEIC_ACID_DIMERS = ['AA', 'AC', 'AG', 'AT', 'AU',
                     'AN', 'CA', 'CC', 'CG', 'CT', 'CU',
                     'CN', 'GA', 'GC', 'GG', 'GT', 'GU',
                     'GN', 'TA', 'TC', 'TG', 'TT', 'TU',
                     'TN', 'UA', 'UC', 'UG', 'UT', 'UU',
                     'UN', 'NA', 'NC', 'NG', 'NT', 'NU',
                     'NN']

NUCLEIC_ACID_TRIMERS = ['ACG', 'ACT', 'ACU', 'ACN', 'AGC', 'AGT',
                     'AGU', 'AGN', 'ATC', 'ATG', 'ATN', 'AUC', 'AUG',
                     'AUN', 'ANC', 'ANG', 'ANT', 'ANU', 'CAG', 'CAT',
                     'CAU', 'CAN', 'CGA', 'CGT', 'CGU', 'CGN', 'CTA',
                     'CTG', 'CTN', 'CUA', 'CUG', 'CUN', 'CNA', 'CNG',
                     'CNT', 'CNU', 'GAC', 'GAT', 'GAU', 'GAN', 'GCA',
                     'GCT', 'GCU', 'GCN', 'GTA', 'GTC', 'GTN', 'GUA',
                     'GUC', 'GUN', 'GNA', 'GNC', 'GNT', 'GNU', 'TAC',
                     'TAG', 'TAN', 'TCA', 'TCG', 'TCN', 'TGA', 'TGC',
                     'TGN', 'TNA', 'TNC', 'TNG', 'UAC', 'UAG', 'UAN',
                     'UCA', 'UCG', 'UCN', 'UGA', 'UGC', 'UGN', 'UNA',
                     'UNC', 'UNG', 'NAC', 'NAG', 'NAT', 'NAU', 'NCA',
                     'NCG', 'NCT', 'NCU', 'NGA', 'NGC', 'NGT', 'NGU',
                     'NTA', 'NTC', 'NTG', 'NUA', 'NUC', 'NUG']

NUCLEIC_ACID_TRIMERS = ['AAA', 'AAC', 'AAG', 'AAT', 'AAU', 'AAN',
                     'ACA', 'ACC', 'ACG', 'ACT', 'ACU', 'ACN',
                     'AGA', 'AGC', 'AGG', 'AGT', 'AGU', 'AGN',
                     'ATA', 'ATC', 'ATG', 'ATT', 'ATU', 'ATN',
                     'AUA', 'AUC', 'AUG', 'AUT', 'AUU', 'AUN',
                     'ANA', 'ANC', 'ANG', 'ANT', 'ANU', 'ANN',
                     'CAA', 'CAC', 'CAG', 'CAT', 'CAU', 'CAN',
                     'CCA', 'CCC', 'CCG', 'CCT', 'CCU', 'CCN',
                     'CGA', 'CGC', 'CGG', 'CGT', 'CGU', 'CGN',
                     'CTA', 'CTC', 'CTG', 'CTT', 'CTU', 'CTN',
                     'CUA', 'CUC', 'CUG', 'CUT', 'CUU', 'CUN',
                     'CNA', 'CNC', 'CNG', 'CNT', 'CNU', 'CNN',
                     'GAA', 'GAC', 'GAG', 'GAT', 'GAU', 'GAN',
                     'GCA', 'GCC', 'GCG', 'GCT', 'GCU', 'GCN',
                     'GGA', 'GGC', 'GGG', 'GGT', 'GGU', 'GGN',
                     'GTA', 'GTC', 'GTG', 'GTT', 'GTU', 'GTN',
                     'GUA', 'GUC', 'GUG', 'GUT', 'GUU', 'GUN',
                     'GNA', 'GNC', 'GNG', 'GNT', 'GNU', 'GNN',
                     'TAA', 'TAC', 'TAG', 'TAT', 'TAU', 'TAN',
                     'TCA', 'TCC', 'TCG', 'TCT', 'TCU', 'TCN',
                     'TGA', 'TGC', 'TGG', 'TGT', 'TGU', 'TGN',
                     'TTA', 'TTC', 'TTG', 'TTT', 'TTU', 'TTN',
                     'TUA', 'TUC', 'TUG', 'TUT', 'TUU', 'TUN',
                     'TNA', 'TNC', 'TNG', 'TNT', 'TNU', 'TNN',
                     'UAA', 'UAC', 'UAG', 'UAT', 'UAU', 'UAN',
                     'UCA', 'UCC', 'UCG', 'UCT', 'UCU', 'UCN',
                     'UGA', 'UGC', 'UGG', 'UGT', 'UGU', 'UGN',
                     'UTA', 'UTC', 'UTG', 'UTT', 'UTU', 'UTN',
                     'UUA', 'UUC', 'UUG', 'UUT', 'UUU', 'UUN',
                     'UNA', 'UNC', 'UNG', 'UNT', 'UNU', 'UNN',
                     'NAA', 'NAC', 'NAG', 'NAT', 'NAU', 'NAN',
                     'NCA', 'NCC', 'NCG', 'NCT', 'NCU', 'NCN',
                     'NGA', 'NGC', 'NGG', 'NGT', 'NGU', 'NGN',
                     'NTA', 'NTC', 'NTG', 'NTT', 'NTU', 'NTN',
                     'NUA', 'NUC', 'NUG', 'NUT', 'NUU', 'NUN',
                     'NNA', 'NNC', 'NNG', 'NNT', 'NNU', 'NNN']

DNA_VOCAB = ['A', 'C', 'G', 'T']

DNA_DIMERS = ['AA', 'AC', 'AG', 'AT',
             'CA', 'CC', 'CG', 'CT',
             'GA', 'GC', 'GG', 'GT',
             'TA', 'TC', 'TG', 'TT']

DNA_TRIMERS = ['AAA', 'AAC', 'AAG', 'AAT',
 'ACA', 'ACC', 'ACG', 'ACT', 'AGA',
 'AGC', 'AGG', 'AGT', 'ATA', 'ATC',
 'ATG', 'ATT', 'CAA', 'CAC', 'CAG',
 'CAT', 'CCA', 'CCC', 'CCG', 'CCT',
 'CGA', 'CGC', 'CGG', 'CGT', 'CTA',
 'CTC', 'CTG', 'CTT', 'GAA', 'GAC',
 'GAG', 'GAT', 'GCA', 'GCC', 'GCG',
 'GCT', 'GGA', 'GGC', 'GGG', 'GGT',
 'GTA', 'GTC', 'GTG', 'GTT', 'TAA',
 'TAC', 'TAG', 'TAT', 'TCA', 'TCC',
 'TCG', 'TCT', 'TGA', 'TGC', 'TGG',
 'TGT', 'TTA', 'TTC', 'TTG', 'TTT']

SELFIES_VOCAB = ['[O]', '[=C]', '[Branch1]', '[#C]', '[N]', '[C]',
 '[Br]', '[Ring1]', '[#Branch1]', '[F]', '[=Branch1]', '[=O]',
 '[#Branch2]', '[C@@H1]', '[Branch2]', '[=N]', '[=Branch2]', '[S]',
 '[=Ring1]', '[Cl]', '[C@H1]', '[NH1]', '[Ring2]', '[C@]',
 '[C@@]', '[P]', '[N+1]', '[O-1]', '[#N]', '[/C]', '[/C@@H1]',
 '[=Ring2]', '[\\Cl]', '[=N+1]', '[/Cl]', '[/S]', '[\\C]',
 '[=S]', '[S@@]', '[S@]', '[/N]', '[I]', '[/O]',
 '[P@]', '[=S@]', '[\\S]', '[\\O]', '[Si]', '[\\N]',
 '[=S@@]', '[/C@H1]', '[/F]', '[\\C@@H1]', '[B]', '[/C@]',
 '[\\F]', '[\\C@@]', '[CH1]', '[\\C@]', '[\\C@H1]', '[CH0]',
 '[=P]', '[/C@@]', '[P@@]']

SELFIES_VOCAB_LEGACY = ['[C]', '[Ring1]', '[=C]', '[Branch1_1]',
             '[N]', '[Branch1_2]', '[=O]', '[O]', '[Branch2_1]',
             '[=N]', '[Ring2]', '[C@Hexpl]', '[C@@Hexpl]', '[F]',
             '[S]', '[Branch1_3]', '[Branch2_2]', '[Branch2_3]', '[#C]',
             '[Expl=Ring1]', '[P]', '[Cl]', '[NHexpl]', '[Br]',
             '[/C]', '[C@expl]', '[C@@expl]', '[#N]', '[O-expl]',
             '[N+expl]', '[Expl=Ring2]', '[\\C]', '[=S]', '[I]',
             '[S@expl]', '[S@@expl]', '[=N+expl]', '[/N]', '[/Cl]',
             '[\\Cl]', '[/O]', '[/S]', '[Siexpl]', '[\\S]',
             '[=S@expl]', '[=S@@expl]', '[\\N]', '[/C@@Hexpl]', '[/C@Hexpl]',
             '[\\O]', '[\\C@Hexpl]', '[\\C@@Hexpl]', '[B]', '[/F]',
             '[/C@expl]', '[\\C@expl]', '[CHexpl]', '[\\F]', '[P@expl]',
             '[Cexpl]', '[/C@@expl]', '[\\C@@expl]', '[=P]', '[P@@expl]',
             '[/NH+expl]', '[/S-expl]', '[=NH+expl]', '[N-expl]', '[NH+expl]',
             '[NH2+expl]', '[NH3+expl]', '[S-expl]', '[\\NHexpl]', '[\\O-expl]',
             '[\\S-expl]']

# includes tokens that appear <500 times in a dataset of 79 million compounds
SELFIES_EXPANDED_VOCAB_LEGACY = ['[O]', '[=C]', '[Branch1]', '[#C]', '[N]', '[C]',
 '[Br]', '[Ring1]', '[#Branch1]', '[F]', '[=Branch1]', '[=O]',
 '[#Branch2]', '[C@@H1]', '[Branch2]', '[=N]', '[=Branch2]', '[S]',
 '[=Ring1]', '[Cl]', '[C@H1]', '[NH1]', '[Ring2]', '[C@]',
 '[C@@]', '[P]', '[N+1]', '[O-1]', '[#N]', '[/C]',
 '[/C@@H1]', '[=Ring2]', '[\\Cl]', '[=N+1]', '[/Cl]', '[/S]',
 '[\\C]', '[=S]', '[S@@]', '[S@]', '[/N]', '[I]',
 '[/O]', '[P@]', '[=S@]', '[\\S]', '[\\O]', '[Si]',
 '[\\N]', '[=S@@]', '[/C@H1]', '[/F]', '[\\C@@H1]', '[B]',
 '[/C@]', '[\\F]', '[\\C@@]', '[CH1]', '[=P@@]', '[\\NH1]',
 '[\\C@]', '[\\C@H1]', '[CH2]', '[Sn]', '[/S@]', '[CH0]',
 '[=P]', '[/C@@]', '[S+1]', '[/NH1]', '[=N-1]', '[/N+1]',
 '[N-1]', '[\\Br]', '[P@@]', '[=S@+1]', '[N@@+1]', '[/Br]',
 '[=P@]', '[/S@@]', '[=O+1]', '[\\N+1]', '[OH0]', '[\\S@]',
 '[N@+1]', '[/B]', '[/I]', '[C-1]', '[CH1-1]', '[-/Ring2]',
 '[\\O-1]', '[/S+1]', '[/OH0]', '[=17O]', '[#N+1]', '[SH1]',
 '[=S+1]', '[B-1]', '[PH1]', '[P@@H1]', '[/O-1]', '[\\I]',
 '[S@@+1]', '[=NH0]', '[I+1]', '[O+1]', '[=P@H1]', '[P+1]',
 '[-/Ring1]', '[\\Sn]', '[-\\Ring2]', '[S@+1]', '[CH2-1]', '[NH0]',
 '[\\S@@]', '[\\C-1]', '[B@-1]', '[\\P@@]', '[=SH1]', '[=S@@+1]',
 '[\\Si]', '[SnH4+2]', '[B@@-1]', '[Sn+1]', '[/P@]', '[\\B]',
 '[=Sn]', '[=P+1]', '[=P@@H1]', '[C+1]', '[\\P@]', '[N@@H1+1]',
 '[/P@@]', '[Sn+3]', '[/Si]', '[/C-1]', '[/CH0]', '[BH3-1]',
 '[\\CH1-1]', '[=B]', '[=Si]', '[/CH1]', '[/Sn]', '[BH2-1]',
 '[\\CH0]', '[\\P]', '[=PH1]']

SELFIES_EXPANDED_VOCAB_LEGACY = ['[C]', '[Ring1]', '[=C]',
             '[Branch1_1]', '[N]', '[Branch1_2]', '[=O]', '[O]', '[Branch2_1]',
             '[=N]', '[Ring2]', '[C@Hexpl]', '[C@@Hexpl]', '[F]', '[S]',
             '[Branch1_3]', '[Branch2_2]', '[Branch2_3]', '[#C]', '[Expl=Ring1]', '[P]',
             '[Cl]', '[NHexpl]', '[Br]', '[/C]', '[C@expl]', '[C@@expl]',
             '[#N]', '[O-expl]', '[N+expl]', '[Expl=Ring2]', '[\\C]', '[=S]',
             '[I]', '[S@expl]', '[S@@expl]', '[=N+expl]', '[/N]', '[/Cl]',
             '[\\Cl]', '[/O]', '[/S]', '[Siexpl]', '[\\S]', '[=S@expl]',
             '[=S@@expl]', '[\\N]', '[/C@@Hexpl]', '[/C@Hexpl]', '[\\O]', '[\\C@Hexpl]',
             '[\\C@@Hexpl]', '[B]', '[/F]', '[/C@expl]', '[\\C@expl]', '[CHexpl]',
             '[\\F]', '[P@expl]', '[Cexpl]', '[/C@@expl]', '[\\C@@expl]', '[=P]',
             '[P@@expl]', '[/Br]', '[=N-expl]', '[/N+expl]', '[S+expl]', '[\\NHexpl]',
             '[\\Br]', '[/NHexpl]', '[N@+expl]', '[/S@expl]', '[N@@+expl]', '[N-expl]',
             '[/S@@expl]', '[CH2expl]', '[=P@expl]', '[Oexpl]', '[Snexpl]', '[\\S@expl]',
             '[C-expl]', '[/B]', '[\\N+expl]', '[#N+expl]', '[=P@@expl]',
             '[/NH+expl]', '[/S-expl]', '[=NH+expl]', '[N-expl]', '[NH+expl]',
             '[NH2+expl]', '[NH3+expl]', '[S-expl]', '[\\NHexpl]', '[\\O-expl]',
             '[\\S-expl]', '[CH-expl]',
             '[\\O-expl]', '[Expl/Ring2]', '[/Oexpl]', '[B-expl]', '[S@@+expl]', '[=S+expl]',
             '[P+expl]', '[/O-expl]', '[PHexpl]', '[=S@+expl]', '[P@@Hexpl]', '[\\I]',
             '[Expl/Ring1]', '[Expl\\Ring2]', '[S@+expl]', '[/I]', '[Nexpl]', '[=B]',
             '[=O+expl]', '[O+expl]', '[CH2-expl]', '[B@-expl]', '[=S@@+expl]', '[B@@-expl]',
             '[\\B]', '[/S+expl]', '[SHexpl]', '[\\S@@expl]', '[\\P@@expl]', '[/P@expl]',
             '[=P@@Hexpl]', '[\\P@expl]', '[/P@@expl]', '[/Siexpl]', '[=17Oexpl]', '[=Nexpl]',
             '[I+expl]', '[=P@Hexpl]', '[\\Snexpl]', '[\\C-expl]', '[=SHexpl]', '[\\Siexpl]',
             '[SnH4+2expl]', '[Sn+expl]', '[=Snexpl]', '[=P+expl]', '[C+expl]', '[N@@H+expl]',
             '[Sn+3expl]', '[/C-expl]', '[/Cexpl]', '[BH3-expl]', '[\\CH-expl]', '[=Siexpl]',
             '[/CHexpl]', '[/Snexpl]', '[BH2-expl]', '[\\Cexpl]', '[\\P]', '[=PHexpl]',
             '[#N+expl]', '[#NH+expl]', '[#PHexpl]', '[#P]', '[#Pexpl]', '[#SHexpl]',
             '[#S]', '[#Sexpl]', '[/Br]', '[/CHexpl]', '[/Cexpl]', '[/N+expl]',
             '[/NHexpl]', '[/O-expl]', '[/PHexpl]', '[/P]', '[/SHexpl]', '[=CHexpl]', '[=Cexpl]',
             '[=N-expl]', '[=P+expl]', '[=P@@expl]', '[=P@expl]', '[=PHexpl]', '[=Pexpl]',
             '[=S-expl]', '[=SHexpl]', '[=Sexpl]', '[=Siexpl]', '[Expl#Ring1]', '[Expl#Ring2]',
             '[Expl/Ring1]', '[Expl/Ring2]', '[Expl\\Ring1]', '[Expl\\Ring2]', '[P+expl]', '[PHexpl]',
             '[Pexpl]', '[SHexpl]', '[Sexpl]', '[\\Br]', '[\\CHexpl]', '[\\Cexpl]',
             '[\\I]', '[\\N+expl]', '[\\SHexpl]', '[\\Siexpl]']


# Cell

def pad_vocab(vocab):
    '''
    pads `vocab` to have a length divisible by 8 - improves fp16 performance
    '''
    if not len(vocab)%8==0:
        final_length = np.ceil(len(vocab)/8)*8
        to_add = len(vocab) - final_length
        vocab = vocab + ['extra']*to_add

    return vocab

# Cell

SMILE_REGEX = """(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|H|\(|\)|\.|=|
                 #|-|\+|\\\\|\/|:|~|@|\?|>|#|\*|\$|\%[0-9]{2}|[0-9])"""

MAPPING_REGEX = """(\[.\*:.]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|H|\[|\]|\(|\)|\.|=|
                    #|-|\+|\\\\|\/|:|~|@|\?|>|#|\*|\$|\%[0-9]{2}|[0-9])"""

AA_MAPPING_REGEX = """(\[.\*:.]|A|C|D|E|F|G|H|I|K|L|M|N|P|Q|R|S|T|V|W|Y)"""

# Cell

def tokenize_by_character(input):
    "Splits `input` into inividual characters"
    unks = False
    if 'unk' in input:
        input = input.replace('unk', '_')
        unks = True
    tokens = [i for i in input]
    if unks:
        for i, item in enumerate(tokens):
            if item=='_':
                tokens[i] = 'unk'
    return tokens

def tokenize_with_replacements(input, replacement_dict):
    "Replaces substrings in `input` using `replacement_dict`, then tokenizes by character"
    for k,v in replacement_dict.items():
        input = input.replace(k,v)
    return [i for i in input]

def regex_tokenize(input, regex):
    'Uses `regex` to tokenize `input`'
    tokens = [token for token in regex.findall(input)]
    return tokens

def tokenize_by_kmer(input, kmer, stride):
    tokens = [input[i:i+kmer] for i in range(0, len(input), stride)]
    if len(tokens[-1]) != kmer:
        tokens = tokens[:-1]
    return tokens

# Cell

class Vocab():
    '''
    Vocab - base vocabulary class

    Inputs:

    - `itos list`: list of tokens in vocabulary

    - `prefunc Optional[Callable]`: function applied to `input` before tokenization

    - `postfunc Optional[Callable]`: function applied to `input` after reconstruction

    '''
    def __init__(self, itos, prefunc=None, postfunc=None):
        self.special_tokens = ['bos', 'eos', 'pad', 'unk']

        self.itos = self.special_tokens + [i for i in itos if not i in self.special_tokens]
        self.stoi = {self.itos[i]:i for i in range(len(self.itos))}
        self.unks = set()
        self.prefunc = prefunc
        self.postfunc = postfunc

    def _tokenize(self, input):
        'Tokenize `input`'
        raise NotImplementedError

    def tokenize(self, input):
        input = self.preprocess(input)
        toks = self._tokenize(input)
        toks = ['bos'] + toks + ['eos']
        return toks

    def join_tokens(self, tokens):
        return ''.join(tokens)

    def preprocess(self, input):
        if self.prefunc is not None:
            input = self.prefunc(input)
        return input

    def postprocess(self, input):
        if self.postfunc is not None:
            input = self.postfunc(input)
        return input

    def numericalize(self, input):
        'Numericalize `input` into integers'
        output = []
        for tok in input:
            if tok in self.stoi.keys():
                output.append(self.stoi[tok])
            else:
                output.append(self.stoi['unk'])
                self.unks.add(tok)
        return output

    def _reconstruct(self, input):
        'Reconstruct `input` into a string'
        output = []
        for item in input:
            item = self.itos[item]
            if item=='eos':
                break

            if (not item=='bos') and (not item=='pad'):
                output.append(item)

        return output

    def reconstruct(self, input):
        tokens = self._reconstruct(input)
        output = self.join_tokens(tokens)
        output = self.postprocess(output)
        return output

    def reconstruct_trajectory(self, input):
        tokens = self._reconstruct(input)
        return [self.join_tokens(tokens[:i]) for i in range(1,len(tokens)+1)]

    def update_vocab(self):
        'Adds tokens in `self.unks` to vocabulary'
        unks = list(self.unks)
        self.itos += unks
        self.stoi = {self.itos[i]:i for i in range(len(self.itos))}
        self.unks = set()

    def update_vocab_from_data(self, inputs):
        'Tokenizes `inputs` and updates the vocabulary with any unknown tokens'
        _ = [self.numericalize(self.tokenize(i)) for i in inputs]
        self.update_vocab()


class CharacterVocab(Vocab):
    '''
    CharacterVocab - tokenize by character

    Inputs:

    - `itos list`: list of tokens in vocabulary

    - `prefunc Optional[Callable]`: function applied to `input` before tokenization

    - `postfunc Optional[Callable]`: function applied to `input` after reconstruction
    '''
    def _tokenize(self, input):
        toks = tokenize_by_character(input)
        return toks

class FuncVocab(Vocab):
    '''
    FuncVocab - tokenize by `tok_func`

    Inputs:

    - `itos list`: list of tokens in vocabulary

    - `tok_func Callable`: tokenization function

    - `prefunc Optional[Callable]`: function applied to `input` before tokenization

    - `postfunc Optional[Callable]`: function applied to `input` after reconstruction
    '''

    def __init__(self, itos, tok_func, prefunc=None, postfunc=None):
        super().__init__(itos, prefunc, postfunc)
        self.tok_func = tok_func

    def _tokenize(self, input):
        toks = self.tok_func(input)
        return toks


class SelfiesVocab(FuncVocab):
    '''
    SelfiesVocab - converts smiles to selfies

    Inputs:

    - `itos list`: list of tokens in vocabulary
    '''
    def __init__(self, itos):
        super().__init__(itos, split_selfie, smile_to_selfie, selfie_to_smile)


class CharacterReplaceVocab(Vocab):
    '''
    CharacterReplaceVocab - tokenize by character with replacement

    Inputs:

    - `itos list`: list of tokens in vocabulary

    - `replace_dict dict`: replacement dictionary of the form
    {multi_character_token : single_character_token}.
    ie replace_dict={'Br':'R', 'Cl':'L'}

    - `prefunc Optional[Callable]`: function applied to `input` before tokenization

    - `postfunc Optional[Callable]`: function applied to `input` after reconstruction

    '''
    def __init__(self, itos, replace_dict, prefunc=None, postfunc=None):
        itos = list(itos)
        self.replace_dict = replace_dict
        if not 'unk' in self.replace_dict.keys():
            self.replace_dict['unk'] = '_'

        self.reverse_dict = {v:k for k,v in replace_dict.items()}
        for rep in self.reverse_dict.keys():
            if not rep in itos:
                itos.append(rep)
        super().__init__(itos, prefunc, postfunc)

    def _tokenize(self, smile):
        toks = tokenize_with_replacements(smile, self.replace_dict)
        return toks

    def _reconstruct(self, input):
        output = []
        for item in input:
            item = self.itos[item]
            if item=='eos':
                break

            if (not item=='bos') and (not item=='pad'):
                if item in self.reverse_dict.keys():
                    item = self.reverse_dict[item]

                output.append(item)

        return output

class RegexVocab(Vocab):
    '''
    RegexVocab - tokenize using `pattern`

    Inputs:

    - `itos list`: list of tokens in vocabulary

    - `pattern str`: regex string

    - `prefunc Optional[Callable]`: function applied to `input` before tokenization

    - `postfunc Optional[Callable]`: function applied to `input` after reconstruction

    '''
    def __init__(self, itos, pattern, prefunc=None, postfunc=None):
        super().__init__(itos, prefunc, postfunc)

        self.pattern = pattern
        self.regex = re.compile(self.pattern)

    def _tokenize(self, smile):
        toks = regex_tokenize(smile, self.regex)
        return toks


class KmerVocab(Vocab):
    '''
    KmerVocab - Kmer tokenization vocabulary

    Inputs:

    - `itos list`: list of tokens in vocabulary

    - `kmer int`: kmer size

    - `stride Optional[int]`: kmer stride. If not passed, stride
    will be the same as kmer. Using a stride value different from
    the kmer value will prevent proper reconstruction

    - `prefunc Optional[Callable]`: function applied to `input` before tokenization

    - `postfunc Optional[Callable]`: function applied to `input` after reconstruction

    '''
    def __init__(self, itos, kmer, stride=None, prefunc=None, postfunc=None):
        super().__init__(itos, prefunc, postfunc)
        self.kmer = kmer

        if stride is None:
            stride = kmer

        self.stride = stride

    def _tokenize(self, input):
        toks = tokenize_by_kmer(input, self.kmer, self.stride)
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