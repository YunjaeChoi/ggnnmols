import argparse
import logging
import pandas as pd
import numpy as np
from rdkit import Chem
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from copy import deepcopy


parser = argparse.ArgumentParser()
parser.add_argument('--file_dir', type=str, default='data/', help='Data file directory')
parser.add_argument('--max_nodes', type=int, default=132, help='Max graph nodes(atoms)')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed')


def validate_mols(mols):
    """
    input: rdkit mols
    ops:
        1. kekulize
        2. remove Hs
        3. sanitize
    returns: valid_mols, valid_index
    """
    valid_mols = []
    valid_index = []
    for i in range(len(mols)):
        mol = mols[i]
        if mol:
            Chem.Kekulize(mol)
            mol = Chem.AddHs(mol)
            mol = Chem.RemoveHs(mol)
            Chem.SanitizeMol(mol)
            valid_index.append(i)
            valid_mols.append(mol)
    return valid_mols, valid_index

def validate_smiles(smiles):
    """
    input: smiles
    ops:
        1. validate_mols
        2. remove duplicates by np.unique
    returns: valid_mols_unique, valid_smiles_unique
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid_mols, valid_index = validate_mols(mols)
    logging.info('Number of valid mols: ' + str(len(valid_mols))
                 + ', Number of discarded mols: ' + str(len(mols) - len(valid_mols)))
    valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
    valid_smiles_unique = np.unique(np.array(valid_smiles))
    valid_mols_unique = [Chem.MolFromSmiles(s) for s in valid_smiles_unique]
    logging.info('Number of valid mols: ' + str(len(valid_mols))
                 + ', Number of unique mols: ' + str(len(valid_mols_unique)))
    return valid_mols_unique, valid_smiles_unique


def main():
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    #tox21 data
    tox21_data = pd.read_csv(args.file_dir + 'tox21_compoundData.csv')
    logging.info('Number of tox21(csv) mols: ' + str(len(tox21_data)))
    tox21_suppl = Chem.SDMolSupplier(args.file_dir + 'tox21.sdf')
    tox21_mols = [x for x in tox21_suppl]
    logging.info('Number of tox21(sdf) mols: ' + str(len(tox21_mols)))
    tox21_valid_mols, tox21_valid_index = validate_mols(tox21_mols)
    logging.info('Number of valid tox21 mols: ' + str(len(tox21_valid_mols))
                 + ', Number of discarded mols: ' + str(len(tox21_mols) - len(tox21_valid_mols)))
    tox21_labels = ['NR.AhR', 'NR.AR', 'NR.AR.LBD', 'NR.Aromatase', 'NR.ER', 'NR.ER.LBD',
                    'NR.PPAR.gamma', 'SR.ARE', 'SR.ATAD5', 'SR.HSE', 'SR.MMP', 'SR.p53']
    tox21_y = tox21_data[tox21_labels]
    tox21_y_valid = np.array(tox21_y)[tox21_valid_index]
    logging.info('Valid tox21 labels shape: ' + str(tox21_y_valid.shape))

    #atoms
    num_atoms_in_mol = [mol.GetNumAtoms() for mol in tox21_valid_mols]
    max_num_atoms_in_mol = np.array(num_atoms_in_mol).max()
    logging.info('Max number of atoms in current mols: ' + str(max_num_atoms_in_mol))
    if max_num_atoms_in_mol > args.max_nodes:
        raise ValueError('max_num_atoms is greater than max_nodes')
    num_atoms_all = Counter()
    for mol in tox21_valid_mols:
        for atom in mol.GetAtoms():
            num_atoms_all[atom.GetAtomicNum()] += 1
    logging.info('Number of all atoms in current mols: ' + str(num_atoms_all))

    #types
    atom_type = np.array(list(set(num_atoms_all.elements())))
    logging.info('Atom types: ' + str(atom_type))
    max_atom_num = atom_type.max()
    logging.info('Max atom number: ' + str(max_atom_num))
    hybridization_type = np.array([Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                          Chem.rdchem.HybridizationType.SP3D2])
    bond_type = np.array([Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])
    num_atom_type = atom_type.shape[0]
    num_hybridization_type = hybridization_type.shape[0]
    num_bond_type = bond_type.shape[0]
    logging.info('Number of atom types: ' + str(num_atom_type))
    logging.info('Number of hybridization types: ' + str(num_hybridization_type))
    logging.info('Number of bond types: ' + str(num_bond_type))
    #dimesions
    #dims := atom_type_onehot: n, atomic_num: 1, formal_charge: 1, radical_electrons:1 ,aromatic:1, hybridization_onehot:5
    dim_concat_atom_feat = num_atom_type + 4 + num_hybridization_type
    #dims := bond_type:4, is in ring:1
    dim_concat_bond_feat = 1 + num_bond_type

    def featurize_mol(mol):
        """
        input: rdkit mols
        ops:
            1. remove Hs
            2. get atom features
            3. get edge features
        returns: atom_feat_matrix, edge_feat_matrix
        """
        mol = Chem.RemoveHs(mol)
        num_atom = mol.GetNumAtoms()

        atom_feat_matrix = np.zeros((num_atom, dim_concat_atom_feat))
        edge_feat_matrix = np.zeros((num_atom, num_atom, dim_concat_bond_feat))

        for i in range(num_atom):
            #atom features
            atom_i = mol.GetAtomWithIdx(i)
            atom_type_onehot = (atom_type == atom_i.GetAtomicNum()).astype(int).reshape(1,-1)
            atomic_num = np.array(atom_i.GetAtomicNum()).reshape(1,-1)
            formal_charge = np.array(atom_i.GetFormalCharge()).reshape(1,-1)
            radical_electrons = np.array(atom_i.GetNumRadicalElectrons()).reshape(1,-1)
            aromatic = np.array(int(atom_i.GetIsAromatic())).reshape(1,-1)
            hybridization_onehot = (hybridization_type == atom_i.GetHybridization()).astype(float).reshape(1,-1)
            atom_features = np.concatenate((atom_type_onehot,atomic_num,formal_charge,
                                           radical_electrons,aromatic,hybridization_onehot),axis=-1)
            atom_feat_matrix[i,:] = atom_features

            #bond features
            for j in range(num_atom):
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond != None:
                    bond_onehot = (bond_type == bond.GetBondType()).astype(float).reshape(1,-1)
                    ring = np.array(bond.IsInRing()).reshape(1,-1)
                    edge_features = np.concatenate((bond_onehot,ring),axis=-1)
                    edge_feat_matrix[i,j,:] = edge_features
                    edge_feat_matrix[j,i,:] = edge_features

        return atom_feat_matrix, edge_feat_matrix

    #for nomalizing formal_charge(atom feat) - idx:55
    formal_charge_divide_val = 3.0

    logging.info('Extracting features...')
    atom_features = np.zeros((len(tox21_valid_mols), args.max_nodes, dim_concat_atom_feat))
    edge_features = np.zeros((len(tox21_valid_mols), args.max_nodes, args.max_nodes, dim_concat_bond_feat))
    for i, mol in enumerate(tqdm(tox21_valid_mols)):
        atom_feat_matrix, edge_feat_matrix= featurize_mol(mol)
        atom_features[i,:atom_feat_matrix.shape[0],:atom_feat_matrix.shape[1]] = atom_feat_matrix
        edge_features[i,:edge_feat_matrix.shape[0],:edge_feat_matrix.shape[1],:edge_feat_matrix.shape[2]] = edge_feat_matrix

    logging.info('Dividing atom num by max atom num')
    atom_features[:,:,54] /= max_atom_num
    logging.info('Dividing formal charge by {}'.format(formal_charge_divide_val))
    atom_features[:,:,55] /= formal_charge_divide_val

    tox21_y_valid_nan = np.isnan(tox21_y_valid)
    tox21_y_valid_notnan = np.zeros_like(tox21_y_valid) + ~tox21_y_valid_nan
    tox21_y_valid[tox21_y_valid_nan] = 0

    validation_idx = 0
    test_idx = 0
    validation_idx_found = False
    test_idx_found = False

    for i in range(len(tox21_valid_index)):
        if (validation_idx_found == False and tox21_data['set'].iloc[tox21_valid_index[i]] == 'validation'):
            validation_idx = i
            validation_idx_found = True
        if (test_idx_found == False and tox21_data['set'].iloc[tox21_valid_index[i]] == 'test'):
            test_idx = i
            test_idx_found = True
    logging.info('tox21 validation idx start: ' + str(validation_idx))
    logging.info('tox21 test idx start: ' + str(test_idx))

    atom_features_tr = atom_features[:validation_idx]
    atom_features_val = atom_features[validation_idx:test_idx]
    atom_features_te = atom_features[test_idx:]

    edge_features_tr = edge_features[:validation_idx]
    edge_features_val = edge_features[validation_idx:test_idx]
    edge_features_te = edge_features[test_idx:]

    y_tr = tox21_y_valid[:validation_idx]
    y_val = tox21_y_valid[validation_idx:test_idx]
    y_te = tox21_y_valid[test_idx:]

    y_notnan_tr = tox21_y_valid_notnan[:validation_idx]
    y_notnan_val = tox21_y_valid_notnan[validation_idx:test_idx]
    y_notnan_te = tox21_y_valid_notnan[test_idx:]
    logging.info('atom features shape: \ntrain: ' + str(atom_features_tr.shape) +
                 '\nvalidation: ' + str(atom_features_val.shape) +
                 '\ntest: ' + str(atom_features_te.shape))
    logging.info('edge features shape: \ntrain: ' + str(edge_features_tr.shape) +
                 '\nvalidation: ' + str(edge_features_val.shape) +
                 '\ntest: ' + str(edge_features_te.shape))
    logging.info('class data shape: \ntrain: ' + str(y_tr.shape) +
                 '\nvalidation: ' + str(y_val.shape) +
                 '\ntest: ' + str(y_te.shape))

    np.savez(args.file_dir + 'tox21_graph',
             atom_feat_tr=atom_features_tr,
             atom_feat_val=atom_features_val,
             atom_feat_te=atom_features_te,
             edge_feat_tr=edge_features_tr,
             edge_feat_val=edge_features_val,
             edge_feat_te=edge_features_te,
             y_tr=y_tr,
             y_val=y_val,
             y_te=y_te,
             y_notnan_tr=y_notnan_tr,
             y_notnan_val=y_notnan_val,
             y_notnan_te=y_notnan_te)

if __name__ == "__main__":
    main()
