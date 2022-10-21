import torch
import sys
from collections import defaultdict
from typing import List, Optional, Union
from multiprocessing import current_process
import warnings

import rdkit
import datamol
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds
from torch_geometric.data import Data
from featurizer.day_light_fg_smarts_list import DAY_LIGHT_FG_SMARTS_LIST

from featurizer.descriptors import (all_descriptors,
                         descriptor_2d_fn_map,
                         descriptor_3d_fn_map,
                         fetch_descriptor_fn)

from featurizer.datamol_conformer import generate as datamol_conf_gen

from featurizer.metadata_2d_descriptors import DESCRIPTORS_METADATA_2D
from featurizer.metadata_3d_descriptors import DESCRIPTORS_METADATA_3D
from featurizer.constants import (MAX_NUM_ATOMS, NUM_DESCRIPTOR_FEATURES_2D, NUM_DESCRIPTOR_FEATURES_3D)

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def rdchem_enum_to_list(values):
    return [values[i] for i in range(len(values))]

def get_atom_feature_dims(list_of_acquired_feature_names):
    if not isinstance(list_of_acquired_feature_names, list):
        list_of_acquired_feature_names = [list_of_acquired_feature_names]
    return list(map(len, [Featurizer.atom_vocab_dict[name] for name in list_of_acquired_feature_names]))

def get_bond_feature_dims(list_of_acquired_feature_names):
    if not isinstance(list_of_acquired_feature_names, list):
        list_of_acquired_feature_names = [list_of_acquired_feature_names]
    list_bond_feat_dim = list(map(len, [Featurizer.bond_vocab_dict[name] for name in list_of_acquired_feature_names]))
    return [l + 1 for l in list_bond_feat_dim]

def get_val(v: pd.Series, dtype):
    if dtype == np.dtype('O'):
        val = np.asarray(v.values.tolist()).astype(np.float32)
    else:
        val = v.values.astype(np.float32)
    return val

def conf_to_mol(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(Chem.Conformer(conf))
    return new_mol

def get_descriptor2d_array(descriptors_2d):
    col_num = 0
    descriptors = np.zeros((len(descriptors_2d), NUM_DESCRIPTOR_FEATURES_2D))
    for k in DESCRIPTORS_METADATA_2D:
        dtype = descriptors_2d[k].dtype
        v = descriptors_2d[k]
        val = get_val(v, dtype)
        l = DESCRIPTORS_METADATA_2D[k]["len"]
        descriptors[:, col_num:col_num + l] = val.reshape(-1, l)
        col_num += l
    return descriptors

def get_descriptor3d_array(descriptors_3d):
    col_num = 0
    descriptors = np.zeros((len(descriptors_3d), NUM_DESCRIPTOR_FEATURES_3D))
    for k in DESCRIPTORS_METADATA_3D:
        dtype = descriptors_3d[k].dtype
        v = descriptors_3d[k]
        val = get_val(v, dtype)
        l = DESCRIPTORS_METADATA_3D[k]["len"]
        descriptors[:, col_num:col_num + l] = val.reshape(-1, l)
        col_num += l
    return descriptors

ogb_atom_features = [
        "atomic_num",
        "chiral_tag",
        "degree",
        "formal_charge",
        "total_numHs",
        "num_radical_e",
        "hybridization",
        "is_aromatic",
        "atom_is_in_ring"
]

ogb_bond_features = [
    "bond_type",
    "bond_stereo",
    "is_conjugated"
]


class Featurizer(object):
    """"several functions taken from GEM-2 Source code: https://arxiv.org/abs/2208.05863 """

    atom_vocab_dict: dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(Chem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
        "hybridization": rdchem_enum_to_list(Chem.HybridizationType.values),
        "is_aromatic": [0, 1],
        'atom_is_in_ring': [0, 1],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }

    ring_vocab_dict = {
        'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }

    atom_vocab_dict = {**atom_vocab_dict, **ring_vocab_dict}


    bond_vocab_dict: dict = {
        "bond_dir": rdchem_enum_to_list(Chem.BondDir.values),
        "bond_type": rdchem_enum_to_list(Chem.BondType.values),
        "is_in_ring": [0, 1],
        'dist_matrix': [i for i in range(10)],
        'bond_stereo': rdchem_enum_to_list(Chem.BondStereo.values),
        'is_conjugated': [0, 1],
        # 'hop_num': list(range(100)),
    }

    # float features
    atom_float_names = ["van_der_waals_radis", 'mass'] #  "partial_charge",
    bond_float_names = ["bond_length", "bond_angle"]

    ### functional groups
    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167
    period_table = rdkit.Chem.GetPeriodicTable()
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]

    MAX_NUM_ATOMS = 51

    def __init__(self,
                 atom_features: Optional[List] = None,
                 bond_features: Optional[List] = None,
                 add_self_loop: bool = False,
                 include_rings: bool = True,
                 include_fingerprints: bool = True,
                 include_daylight_fg: bool = True,
                 generate_conformer: bool = False,
                 num_conformers: int = 5,
                 generate_descriptor2d: bool = False,
                 generate_descriptor3d: bool = False,
                 conformer_memmap: Optional[np.memmap] = None,
                 descriptor2d_memmap: Optional[np.memmap] = None,
                 descriptor3d_memmap: Optional[np.memmap] = None,
                 num_threads=1,
                 generate_additional_conformers_for_sdf: bool = False
                #  *args,
                #  **kwargs,
                 ):

        """ extract a series of features from smiles strings

        Args:
            atom_features (list): list of atom features to extract. See `Featurizer.list_atom_features()`
            bond_features (list): list of edge features to extract. See `Featurizer.list_bond_features()`
            add_self_loop (bool): add self loops to edges for each atom
            generate_conformer (bool): flag to generate conformers
            num_conformer (int): number of conformers to generate
            generate_descriptor2d (bool): flag to generate descriptor2ds
            generate_descriptor3d (bool): flag to generate descriptor3ds
        """

        if atom_features is None:
            atom_features = list(Featurizer.atom_vocab_dict.keys()) + Featurizer.atom_float_names
        self.atom_features = atom_features
        self.ring_feats =  list(Featurizer.ring_vocab_dict.keys())
        self.add_self_loop = add_self_loop
        if bond_features is None:
            bond_features = list(Featurizer.bond_vocab_dict.keys())
        self.bond_features = bond_features
        self.include_rings = include_rings #any(['in_num_ring_with' in af for af in self.atom_features])
        self.include_fingerprints = include_fingerprints
        self.include_daylight_fg = include_daylight_fg
        self.generate_conformer = generate_conformer
        self.generate_descriptor2d = generate_descriptor2d
        self.generate_descriptor3d = generate_descriptor3d
        self.num_conformers = num_conformers
        self.conformer_memmap = conformer_memmap
        self.descriptor2d_memmap = descriptor2d_memmap
        self.descriptor3d_memmap = descriptor3d_memmap
        self.num_threads = num_threads
        self.generate_additional_conformers_for_sdf = generate_additional_conformers_for_sdf

    def to_mol(self, item):
        if isinstance(item, str): # - assume it's a smile str..
            return Chem.MolFromSmiles(item)
        else: # - assume it's already a mol
            return item

    def get_atom_features(self):
        return self.atom_features + Featurizer.atom_float_names

    def get_bond_features(self):
        return self.bond_features

    @staticmethod
    def get_descriptors2d_features():
        return list(DESCRIPTORS_METADATA_2D.keys())

    @staticmethod
    def get_descriptors3d_features():
        return list(DESCRIPTORS_METADATA_3D.keys())

    @staticmethod
    def get_descriptors2d_metadata():
        return DESCRIPTORS_METADATA_2D

    @staticmethod
    def get_descriptors3d_metadata():
        return DESCRIPTORS_METADATA_3D

    def smiles_to_graph(self,
                        batch: Optional[Union[List[Chem.rdchem.Mol],
                            Chem.rdchem.Mol, List[str], str]],
                        #batch_mol_3d: Optional[ = None,
                        start_idx: Optional[int] = 0,
                        end_idx: Optional[int] = None,
                        ):


        if not isinstance(batch, list):
            batch = [batch]
        if isinstance(batch[0], str):
            smile_mode = True
        else:
            smile_mode = False

        all_data = []

        if end_idx is not None:
            # - assumes batch includes all items [0, total_items]
            dl = range(start_idx, end_idx)
        else:
            dl = range(0, len(batch))

        if (current_process()._identity) and \
            current_process()._identity[0] == 1 and \
            len(batch) > 1:
            dl = tqdm(dl)

        for idx in dl:
            mol = self.to_mol(batch[idx])
            data = defaultdict(list)
            if len(mol.GetAtoms()) == 0:
                continue
            # - atom features
            if self.include_rings:
                ring_list = Featurizer.get_ring_size(mol)
            for i, atom in enumerate(mol.GetAtoms()):
                feat_dict = self.atom_to_feature_vector(atom, self.atom_features)
                # feat_dict = self.atom_to_feature_vector_all(atom, ring_list[i])
                atom_feat = [v for _, v in feat_dict.items()]
                data["x"].append(atom_feat)
                if self.include_rings:
                    ring_dict = self.atom_to_feature_vector(atom, self.ring_feats, ring_list[i])
                    ring_feat = [v for _, v in ring_dict.items()]
                    data["ring_sizes"].append(ring_feat)
            # - edge features
            N = i + 1
            data["num_atoms"] = N
            if len(mol.GetBonds()) > 0: # mol has bonds
                edges_list = []
                bond_features_list = []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    # - add edges in both directions
                    data["edge_index"] += [(i, j), (j, i)]
                    bond_feature = []
                    for bond_feat_name in self.bond_features:
                        bond_feature.append(self.bond_to_feature_vector(bond, bond_feat_name))
                    data["edge_attr"].append(bond_feature)
                    data["edge_attr"].append(bond_feature)

                if self.add_self_loop:
                    for i in range(N):
                        data["edge_index"] += [(i, i)]
                    bond_ids = []
                    for bond_feat_name in self.bond_features:
                        bond_ids.append(get_bond_feature_dims(bond_feat_name)[0] - 1)
                    data["edge_attr"] += [bond_ids] * N
                data["edge_index"] = np.array(data["edge_index"], dtype=np.int32).T
            else:   # mol has no bonds
                bond_feature = np.empty((0, len(self.bond_features)), dtype=np.int32)
                data["edge_index"] = np.empty((2, 0), dtype=np.int32)

            # data["dist_matrix"] = np.array(rdkit.Chem.rdmolops.GetDistanceMatrix(mol)).reshape(1, -1)
            if self.include_rings:
                data["ring_sizes"] = np.array(data["ring_sizes"], dtype=np.int32)
            data["edge_attr"] = np.array(data["edge_attr"], dtype=np.int32)
            data["x"] = np.array(data["x"], dtype=np.int32)

            if self.include_fingerprints:
                morgan_fp = np.array(Featurizer.get_morgan_fingerprint(mol), np.int32).reshape(1, -1)
                maccs_fp = np.array(Featurizer.get_maccs_fingerprint(mol), np.int32).reshape(1, -1)
                data["fingerprint"] = np.concatenate([morgan_fp, maccs_fp], axis=1)

            if self.include_daylight_fg:
                data['daylight_fg_counts'] = np.array(Featurizer.get_daylight_functional_group_counts(mol), np.int32).reshape(1, -1)

            # - descriptors 2d
            if self.generate_descriptor2d:
                df_2 = datamol.descriptors.batch_compute_many_descriptors([mol],
                                                                        properties_fn=descriptor_2d_fn_map,
                                                                        add_properties=False,
                                                                        batch_size=1,
                                                                        n_jobs=self.num_threads)
                descriptors_2d_val = get_descriptor2d_array(df_2)
                data["descriptor_2d"] = descriptors_2d_val
                cols = df_2.columns

            elif self.descriptor2d_memmap is not None:
                data["descriptor_2d"] = self.descriptor2d_memmap[start_idx + idx]

            new_mol = mol
            if self.generate_conformer or self.generate_additional_conformers_for_sdf:
                # - generate conformer
                num_conformers_to_gen = self.num_conformers
                conformer_positions = []
                if not smile_mode:
                    num_conformers_to_gen = self.num_conformers - 1 # - first one will be the original conformers pos
                if num_conformers_to_gen and (smile_mode or self.generate_additional_conformers_for_sdf):
                    try:
                        new_mol, energy = datamol_conf_gen(mol,
                                                        add_hs=True,
                                                        minimize_energy=True,
                                                        n_confs=num_conformers_to_gen,
                                                        forcefield="MMFF94s",
                                                        energy_iterations=100,
                                                        ignore_failure=True,
                                                        num_threads=self.num_threads)
                    except Exception as e:
                        # print(f"Exception occured: {e}, falling back to 2D coordinates")
                        new_mol = mol
                        AllChem.Compute2DCoords(new_mol)
                        energy = [0] * self.num_conformers
                    energy = np.asarray(energy)
                else:
                    new_mol = mol
                if not smile_mode and (self.num_conformers == 0):
                    self.num_conformers = 1
                conformer_positions += [Featurizer.get_atom_poses(mol, conf) for conf in new_mol.GetConformers()]
                conformer_positions_ = np.zeros((N, self.num_conformers, 3))
                conformer_positions = np.stack(conformer_positions).transpose(1, 0, 2)
                conformer_positions_[:, :conformer_positions.shape[1], :] = conformer_positions
                data["conformer_pos"] = conformer_positions_

            elif self.conformer_memmap is not None:
                data["conformer_pos"] = self.conformer_memmap[start_idx + idx, :, :N, :].transpose(1, 0)

            # - descriptors 3d
            if self.generate_descriptor3d:
                batch_mol = [conf_to_mol(new_mol, conf.GetId()) for conf in new_mol.GetConformers()]
                if len(batch_mol):
                    df_3 = datamol.descriptors.batch_compute_many_descriptors(batch_mol,
                                                                    properties_fn=descriptor_3d_fn_map,
                                                                    add_properties=False,
                                                                    batch_size=len(batch_mol),
                                                                    n_jobs=self.num_threads)
                    descriptors_3d_val = get_descriptor3d_array(df_3)
                    data["descriptors_3d"] = descriptors_3d_val
            elif self.descriptor3d_memmap is not None:
                data["descriptor_3d"] = self.descriptor3d_memmap[start_idx + idx]

            all_data.append(data)
        return all_data

    @staticmethod
    def check_partial_charge(atom):
        pc = atom.GetDoubleProp('GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        return pc

    def atom_to_feature_vector_all(self, atom, ring_list):
        feat = {
            "atomic_num": safe_index(Featurizer.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()), # can add +1 for 0.0V emb to be represented by 0 value,
            "chiral_tag": safe_index(Featurizer.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()), # can add +1 for 0.0V emb to be represented by 0 value,
            "degree": safe_index(Featurizer.atom_vocab_dict["degree"], atom.GetTotalDegree()), # can add +1 for 0.0V emb to be represented by 0 value,
            "formal_charge": safe_index(Featurizer.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()), # can add +1 for 0.0V emb to be represented by 0 value,
            "total_numHs": safe_index(Featurizer.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()), # can add +1 for 0.0V emb to be represented by 0 value,
            'num_radical_e': safe_index(Featurizer.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()), # can add +1 for 0.0V emb to be represented by 0 value,
            "hybridization": safe_index(Featurizer.atom_vocab_dict["hybridization"], atom.GetHybridization()), # can add +1 for 0.0V emb to be represented by 0 value,
            "is_aromatic": safe_index(Featurizer.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())), # can add +1 for 0.0V emb to be represented by 0 value,
            'atom_is_in_ring': safe_index(Featurizer.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())), # can add +1 for 0.0V emb to be represented by 0 value,
            "explicit_valence": safe_index(Featurizer.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()), # can add +1 for 0.0V emb to be represented by 0 value,
            "implicit_valence": safe_index(Featurizer.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()), # can add +1 for 0.0V emb to be represented by 0 value,
            'valence_out_shell': safe_index(Featurizer.atom_vocab_dict['valence_out_shell'],
                                            Featurizer.period_table.GetNOuterElecs(atom.GetAtomicNum())), # can add +1 for 0.0V emb to be represented by 0 value,
            'van_der_waals_radis': Featurizer.period_table.GetRvdw(atom.GetAtomicNum()),
            # 'partial_charge': Featurizer.check_partial_charge(atom),
            'mass': atom.GetMass() * 0.01,
             'in_num_ring_with_size3': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size3'], ring_list[0]), # can add +1 for 0.0V emb to be represented by 0 value,
            'in_num_ring_with_size4': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size4'], ring_list[1]), # can add +1 for 0.0V emb to be represented by 0 value,
            'in_num_ring_with_size5': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size5'], ring_list[2]), # can add +1 for 0.0V emb to be represented by 0 value,
            'in_num_ring_with_size6': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size6'], ring_list[3]), # can add +1 for 0.0V emb to be represented by 0 value,
            'in_num_ring_with_size7': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size7'], ring_list[4]), # can add +1 for 0.0V emb to be represented by 0 value,
            'in_num_ring_with_size8': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size8'], ring_list[5]), # can add +1 for 0.0V emb to be represented by 0 value,
        }
        return feat

    def atom_to_feature_vector(self, atom, feature_names, ring_list: Optional[List] = None):
        feats = {}
        for feature_name in feature_names:
            if 'in_num_ring_with' in feature_name:
                feat_val = ring_list[int(feature_name[-1]) - 3]
            else:
                feat_val = Featurizer.get_atom_feature(atom, feature_name)

            feat = safe_index(Featurizer.atom_vocab_dict[feature_name],
                              feat_val)
            feats[feature_name] = feat
        return feats

    def bond_to_feature_vector(self, bond, name):
        return safe_index(Featurizer.bond_vocab_dict[name], Featurizer.get_bond_value(bond, name))

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_atom_poses(mol, conf):
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_atom_feature_id(atom, name):
        assert name in Featurizer.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(Featurizer.atom_vocab_dict[name], Featurizer.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        assert name in Featurizer.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(Featurizer.atom_vocab_dict[name])

    @staticmethod
    def smiles_to_mol(smiles_string, package="rdkit"):
        if package == "rdkit":
            return Chem.MolFromSmiles(smiles_string)
        elif package == "datamol":
            return datamol.to_mol(smiles_string, sanitize=True)
        else:
            raise ValueError(f"package is not supported: {package}")

    @staticmethod
    def canonical_smiles(smiles: str) -> str:
        """canonical smiles string https://github.com/rdkit/rdkit/issues/2747"""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol)
        else: # - rdkit cannot handle str
            return None
    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = Featurizer.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts"""
        fg_counts = []
        for fg_mol in Featurizer.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = Featurizer.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    @staticmethod
    def list_atom_features():
        return list(Featurizer.atom_vocab_dict.keys())

    @staticmethod
    def list_bond_features():
        return list(Featurizer.bond_vocab_dict.keys())

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts"""
        fg_counts = []
        for fg_mol in Featurizer.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts
    @staticmethod
    def get_fragment(mol):
        pass
    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list"""
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)
            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def get_atom_feature(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass()) * 0.01
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':
            return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        elif name == 'valence_out_shell':
            return Featurizer.period_table.GetNOuterElecs(atom.GetAtomicNum())
        elif name == 'van_der_waals_radis':
             return Featurizer.period_table.GetRvdw(atom.GetAtomicNum())
        else:
            raise ValueError(f"{name} not supported")

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            # raise ValueError(f"{name} not supported")
            return None
