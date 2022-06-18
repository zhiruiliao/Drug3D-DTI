import numpy as np
from gensim.models import Word2Vec
import os
from rdkit import Chem
from rdkit.Chem import AllChem

num_atom_feat = 34

def protein_seq_to_idx_mask(seq, vocab_dict, max_len=800):
    if len(seq) > max_len:
        seq = seq[:max_len]
    
    token = np.zeros(max_len, dtype=np.float32)
    mask = np.ones(max_len, dtype=np.float32)
    for i, char in enumerate(seq):
        idx = vocab_dict.get(char, 0)
        token[i] = idx
    mask[: len(seq)] = 0
    mask[np.newaxis, np.newaxis, :]
    return token, mask


VDW_RADIUS = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
              'F': 1.47, 'P': 1.8, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98}

def get_vdw_radii(symbol):
    try:
        return VDW_RADIUS[symbol]
    except KeyError:
        return 1.7


def vanderWaals(d, r):
    if d >= 0 and d < r:
        return np.exp(-2 * d * d / (r * r))
    elif d >= r and d < 1.5 * r:
        return max(d * d / (r * r) * 0.541 - d / r * 1.624 + 1.218, 0)
    else:
        return 0


ATOM_TYPES = dict(P=1, C=2, N=3, O=4, F=5, S=6, Cl=7, Br=8, I=9)
HYBRID_TYPES = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5
}

DEFAULT_MAX_ATOM_NUM = 64
# 9 known atom types + 1 unknown type --> 10
# 5 known hybridization types + 1 unknown type --> 6
# degree --> 7
# formal charge --> 1
# radical electrons --> 1
# aromatic states --> 1
# explicit hydrogen --> 5
# chirality --> 3
# dimension == 10 + 6 + 7 + 1 + 1 + 1 + 5 + 3
# dimension == 34
FEATURE_DIM = 34


def get_atom_features(atom):
    
    one_hot = np.zeros(shape=(FEATURE_DIM,), dtype=np.float32)
    
    atom_type_idx = ATOM_TYPES.get(atom.GetSymbol(), 0)
    one_hot[atom_type_idx] = 1
    
    hybrid_type_idx = HYBRID_TYPES.get(atom.GetHybridization(), 0)
    one_hot[10 + hybrid_type_idx] = 1
    
    degree = atom.GetDegree()
    one_hot[16 + degree] = 1
    
    if atom.GetFormalCharge():
        one_hot[23] = 1
    
    if atom.GetNumRadicalElectrons():
        one_hot[24] = 1
        
    if atom.GetIsAromatic():
        one_hot[25] = 1
    
    explicit_h = min(atom.GetTotalNumHs(), 4)
    one_hot[26 + explicit_h] = 1
    
    if atom.HasProp("_ChiralityPossible"):
        one_hot[31] = 1
        try:
            if atom.GetProp('_CIPCode') == 'S':
                one_hot[32] = 1
            else:
                one_hot[33] = 1
        except:
            one_hot[32] = 1
            one_hot[33] = 1
    else:
        one_hot[31:] = 0
    
    return one_hot


def get_spatial_matrix(atom_rxyz):
    n = atom_rxyz.shape[0]
    sp_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = np.sqrt(np.sum(np.square(atom_rxyz[i, 1:] - atom_rxyz[j, 1:])))
            r = (atom_rxyz[i, 0] + atom_rxyz[j, 0]) / 2
            vdw = vanderWaals(d, r)
            sp_mat[i][j] = vdw
            sp_mat[j][i] = vdw
    return sp_mat


def get_mol_features(smiles: str, max_atom_num=DEFAULT_MAX_ATOM_NUM):

    mol = Chem.MolFromSmiles(smiles)
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mol = Chem.RemoveHs(mol)
    
    if mol is None:
        print(f"SMILES {smiles} invalid")
        return None, None, None, None

    atom_num = mol.GetNumAtoms()
    if atom_num > max_atom_num:
        print(f"SMILES {smiles} is too large")
        return None, None, None, None

    x = np.zeros(shape=(max_atom_num, FEATURE_DIM), dtype=np.float32)
    a = np.zeros(shape=(max_atom_num, max_atom_num), dtype=np.float32)
    s = np.zeros(shape=(max_atom_num, max_atom_num), dtype=np.float32)
    
    mask = np.ones(max_atom_num, dtype=np.float32)
    mask[:min(atom_num, max_atom_num)] = 0
    mask = mask[np.newaxis, np.newaxis, :]
    
    atom_rxyz = np.zeros((atom_num, 4), dtype=np.float32)
    
    try:
        mol.GetConformer()
    except ValueError:
        return None, None, None, None
    
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        x[atom_idx, :] = get_atom_features(atom)
        atom_rxyz[atom_idx, 0] = get_vdw_radii(atom.GetSymbol())
        atom_rxyz[atom_idx, 1:] = mol.GetConformer().GetAtomPosition(atom_idx)
        
    a_ = Chem.GetAdjacencyMatrix(mol).astype(np.float32) + np.eye(atom_num, dtype=np.float32)
    a_degree = np.diag(np.power(np.sum(a_, axis=1), -0.5))
    a_degree[np.isinf(a_degree)] = 0
    a[:atom_num, :atom_num] = np.matmul(np.matmul(a_degree, a_), a_degree)
    
    s_ = get_spatial_matrix(atom_rxyz).astype(np.float32)
    s_degree = np.diag(np.power(np.sum(s_, axis=1), -0.5))
    s_degree[np.isinf(s_degree)] = 0
    s[:atom_num, :atom_num] = np.matmul(np.matmul(s_degree, s_), s_degree)
    
    return x, a, s, mask



def get_token_and_mask(smiles, vocab_dict, max_len=80):
    if len(smiles) > max_len:
        return None, None
    
    token = np.zeros(max_len, dtype=np.float32)
    mask = np.ones(max_len, dtype=np.float32)
    for i, char in enumerate(smiles):
        idx = vocab_dict.get(char, 0)
        token[i] = idx
    mask[: len(smiles)] = 0
    mask[np.newaxis, np.newaxis, :]
    return token, mask


def main_davis(txt_file):
    f = open(txt_file, 'r')
    raw_data = f.read().strip().split('\n')
    f.close()
    
    good_data = []
    for row in raw_data:
        smiles, sequence, label = row.strip().split(',')
        if '.' not in smiles:
            good_data.append([smiles, sequence, label])
    
    f = open("davis_vocab.txt")
    vocab_data = f.read().strip().split('\n')
    f.close()
    vocab = {}
    for row in vocab_data:
        char, idx = row.split(',')
        vocab[char] = int(idx)
    print(vocab)
    vocab_size = len(vocab)
    
    
    f = open("davis_vocab_protein.txt")
    vocab_data = f.read().strip().split('\n')
    f.close()
    vocab_protein = {}
    for row in vocab_data:
        char, idx = row.split(',')
        vocab_protein[char] = int(idx)
    print(vocab)
    
    
    
    
    
    x_list, a_list, s_list, drug_mask_list = [], [], [], []
    protein_list, protein_mask_list = [], []
    y_list = []
    token_list, token_mask_list = [], []
    ecfp_list = []
    
    
    total = len(good_data)
    for i, point in enumerate(good_data):
        print(f"{i + 1} / {total}...")
        smiles, sequence, label = point
        x, a, s, d_mask = get_mol_features(smiles)
        token, token_mask = get_token_and_mask(smiles, vocab)
        mol = Chem.MolFromSmiles(smiles)
        if x is not None and token is not None:
            x_list.append(x)
            a_list.append(a)
            s_list.append(s)
            drug_mask_list.append(d_mask)
        
            protein_token, p_mask = protein_seq_to_idx_mask(sequence, vocab_protein)
            protein_list.append(protein_token)
            protein_mask_list.append(p_mask)
            
            y_list.append(np.array([float(label)]))
            
            token_list.append(token)
            token_mask_list.append(token_mask)
            
            ecfp = np.array(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024).ToBitString()), dtype=np.float32)
            ecfp_list.append(ecfp)
            
    x_list = np.array(x_list, dtype=np.float32)
    a_list = np.array(a_list, dtype=np.float32)
    s_list = np.array(s_list, dtype=np.float32)
    drug_mask_list = np.array(drug_mask_list, dtype=np.float32)
    
    protein_list = np.array(protein_list, dtype=np.float32)
    protein_mask_list = np.array(protein_mask_list, dtype=np.float32)
    
    y_list = np.array(y_list, dtype=np.float32)
    
    token_list = np.array(token_list, dtype=np.float32)
    token_mask_list = np.array(token_mask_list, dtype=np.float32)
    
    ecfp_list = np.array(ecfp_list, dtype=np.float32)
    
    print(x_list.shape, a_list.shape, s_list.shape, drug_mask_list.shape)
    print(protein_list.shape, protein_mask_list.shape)
    print(y_list.shape)
    print(token_list.shape, token_mask_list.shape)
    print(ecfp_list.shape, ecfp_list.dtype)
    
    
    print(x_list[:2], a_list[:2], s_list[:2], drug_mask_list[:2])
    print(protein_list[:2], protein_mask_list[:2])
    print(y_list[:2])
    print(token_list[:2], token_mask_list[:2])
    
    
    np.save(f"{txt_file}_x.npy", x_list)
    np.save(f"{txt_file}_a.npy", a_list)
    np.save(f"{txt_file}_s.npy", s_list)
    np.save(f"{txt_file}_drug_mask.npy", drug_mask_list)
    
    np.save(f"{txt_file}_prot", protein_list)
    np.save(f"{txt_file}_prot_mask.npy", protein_mask_list)
    np.save(f"{txt_file}_y.npy", y_list)
    
    np.save(f"{txt_file}_token.npy", token_list)
    np.save(f"{txt_file}_token_mask.npy", token_mask_list)
    np.save(f"{txt_file}_ecfp.npy", ecfp_list)

main_davis("davis_kinase.csv")




