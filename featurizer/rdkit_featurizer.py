import sys
from collections import defaultdict
from typing import List, Optional, Union

import rdkit
import datamol
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .descriptors import (all_descriptors,
                         descriptor_2d_fn_map,
                         descriptor_3d_fn_map,
                         fetch_descriptor_fn)
from .datamol_conformer import generate as datamol_generate

from .metadata_2d_descriptors import DESCRIPTORS_METADATA_2D
from .metadata_3d_descriptors import DESCRIPTORS_METADATA_3D
from .day_light_fg_smarts_list import DAY_LIGHT_FG_SMARTS_LIST

from .comenet_feature import get_comenet_feature


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
    return list(map(len, [Featurizer.atom_vocab_dict[name] for name in list_of_acquired_feature_names]))

def get_bond_feature_dims(list_of_acquired_feature_names):
    list_bond_feat_dim = list(map(len, [Featurizer.bond_vocab_dict[name] for name in list_of_acquired_feature_names]))
    return list_bond_feat_dim

ogb_atom_features = {
        "atomic_num",
        "chirality",
        "degree",
        "formal_charge",
        "num_h",
        "num_rad_e",
        "hybridization",
        "is_aromatic",
        "is_in_ring"
}

ogb_bond_features = {
    "possible_bond_type_list",
    "possible_bond_stereo_list",
    "possible_is_conjugated_list"
}

class Featurizer(object):
    """"several functions taken from GEM-2 Source code: https://arxiv.org/abs/2208.05863 """

    atom_vocab_dict: dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(Chem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": rdchem_enum_to_list(Chem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
        'atom_is_in_ring': [0, 1],
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }


    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(Chem.BondDir.values),
        "bond_type": rdchem_enum_to_list(Chem.BondType.values),
        "is_in_ring": [0, 1],
        'dist_matrix': [i for i in range(10)],
        'bond_stereo': rdchem_enum_to_list(Chem.BondStereo.values),
        'is_conjugated': [0, 1],
        'hop_num': list(range(100)),
    }

    # float features
    atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']
    bond_float_feats= ["bond_length", "bond_angle"]

    ### functional groups
    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167
    period_table = rdkit.Chem.GetPeriodicTable()
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]

    MAX_NUM_ATOMS = 51
    NUM_DESCRIPTOR_FEATURES_2D = 409
    NUM_DESCRIPTOR_FEATURES_3D = 301

    def __init__(self,
                 atom_features: Optional[List] = None,
                 bond_features: Optional[List] = None,
                 descriptor_features: Optional[List] = None,
                 add_self_loop: bool = False,
                 add_comenet_features: bool = False,
                 num_conformers: int = 0,
                 conformer_package: str = "rdkit",
                 smiles_to_mol_package: str = "rdkit",
                 add_hs: bool = True,
                 skip_dihedral: bool = False,
                 skip_fragment_smiles: bool = False,
                 num_threads: int = 0,
                 *args,
                 **kwargs,
                 ):

        """ extract a series of features from smiles strings

        Args:
            atom_features (list): list of atom features to extract. See `Featurizer.list_atom_features()`
            bond_features (list): list of edge features to extract. See `Featurizer.list_bond_features()`
            descriptor_features (list): list of descriptors to extract. See `Featurizer.list_descriptor_features()`
            add_self_loop (bool): flag for adding self loop edges
            num_conformers (int): number of conformers to generate
            conformer_package (str): package to use for generating conformers either one of ['rdkit', 'datamol'].
            'datamol' conformer generation is much more extensive, 'rdkit' conformer generation here is a simple call to
            `AllChem.EmbedMultipleConf` without any post/pre processing.
            smiles_to_mol_package (str): package to use for converting smiles to mol one of ['rdkit', 'datamol']. 'datamol' does
            some extra sanitization steps.
            add_hs (bool): whether to add hydrogen atoms prior to conformer generation or energy calculation.
            skip_dihedral (bool): flag to skip dihedral mols.
            skip_fragment_smiles (bool): flag to skip smiles with fragments
            num_workers (int): number of workers
        """

        if atom_features is None:
            atom_features = list(Featurizer.atom_vocab_dict.keys())
        self.atom_features = atom_features

        self.add_self_loop = add_self_loop
        if bond_features is None:
            bond_features = list(Featurizer.bond_vocab_dict.keys())
        self.bond_features = bond_features

        self.descriptor_features_map = None
        if descriptor_features is not None:
            self.descriptor_features_map = {feature_name: fetch_descriptor_fn(feature_name)
                                            for feature_name in descriptor_features}

        self.num_conformers = num_conformers
        self.conformer_package = conformer_package
        self.add_hs = add_hs
        self.skip_dihedral = skip_dihedral
        self.skip_fragment_smiles = skip_fragment_smiles
        self.num_threads = num_threads
        self.include_rings = any(['in_num_ring_with' in af for af in self.atom_features])
        self.add_comenet_features = add_comenet_features

    def smiles_to_graph(self,
                        smiles,
                        mol_3d: Optional[Union[List[Chem.rdchem.Mol], Chem.rdchem.Mol]]=None):
        if isinstance(smiles, str):
            smiles = [smiles]
        if mol_3d is not None:
            if not isinstance(mol_3d, list):
                mol_3d = [mol_3d]
        mol_batch = []
        for i, smile in enumerate(smiles):
            # if self.skip_fragment_smiles and Featurizer.fragment_check(smile):
            #     continue
            mol = Chem.MolFromSmiles(smile)#Featurizer.smiles_to_mol(smiles)
            if len(mol.GetAtoms()) == 0:
                continue
            mol_batch.append(mol)

        if not len(mol_batch):
            return None
        batch_data = []
        for mol in mol_batch:
            # if self.skip_dihedral and Featurizer.dihedral_check(mol):
            #     continue
            data = defaultdict(list)
            # - atom features
            ring_list = Featurizer.get_ring_size(mol)
            for i, atom in enumerate(mol.GetAtoms()):
                feat_dict = self.atom_to_feature_vector_all(atom, ring_list[i])
                for k, v in feat_dict.items():
                    data[k].append(v)

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
                    data["edges"] += [(i, j), (j, i)]
                    for bond_feat_name in self.bond_features:
                        bond_feature = self.bond_to_feature_vector(bond, bond_feat_name)
                        data[bond_feat_name] += [bond_feature] * 2

            else:   # mol has no bonds
                for bond_feat_name in self.bond_features:
                    data[bond_feat_name] = np.zeros((0, ), dtype=np.int64)

                data["edges"] = np.zeros((0, 2), dtype=np.int64)

            data["dist_matrix"] = np.asarray(rdkit.Chem.rdmolops.GetDistanceMatrix(mol))
            if self.add_self_loop:
                for i in range(N):
                    data["edges"] += [(i, i)]
                for bond_feat_name in self.bond_features:
                    bond_id = get_bond_feature_dims(bond_feat_name)[0] + 1 + 1
                    data[bond_feat_name] += [bond_id] * N

            data["edges"] = np.array(data["edges"], np.int64)
             ### morgan fingerprint
            data['morgan_fp'] = np.array(Featurizer.get_morgan_fingerprint(mol), 'int64')
            data['maccs_fp'] = np.array(Featurizer.get_maccs_fingerprint(mol), 'int64')
            data['daylight_fg_counts'] = np.array(Featurizer.get_daylight_functional_group_counts(mol), 'int64')
            batch_data.append(data)
        # - descriptors
        df_2 = datamol.descriptors.batch_compute_many_descriptors(mol_batch,
                                                                properties_fn=descriptor_2d_fn_map,
                                                                add_properties=False,
                                                                batch_size=len(mol_batch),
                                                                n_jobs=self.num_threads)
        cols = df_2.columns
        for i in range(len(batch_data)):
            mol_descriptor = df_2.iloc[i]
            for col in cols:
                batch_data[i][col] = mol_descriptor[col]
        new_mol_batch = []
        num_conformers = 5
        for i, mol in enumerate(mol_batch):
            try:
                new_mol, energy = datamol_generate(mol,
                                            add_hs=True,
                                            n_confs=num_conformers,
                                            energy_iterations=100,
                                            num_threads=self.num_threads)
            except Exception as e:
                print(f"Exception occured: {e}, falling back to 2D coordinates")
                new_mol = mol
                AllChem.Compute2DCoords(new_mol)
                energy = [0] * num_conformers

            new_mol_batch.append(new_mol)
            energy = np.asarray(energy)
            conformer_positions_ = np.zeros((num_conformers, Featurizer.MAX_NUM_ATOMS, 3))
            conformer_positions = [Featurizer.get_atom_poses(mol, conf) for conf in new_mol.GetConformers()]
            conformer_positions = np.stack(conformer_positions)
            conformer_positions_[:conformer_positions.shape[0], :conformer_positions.shape[1], :] = conformer_positions
            batch_data[i]["conf_pos"] = conformer_positions
            # - comenet features
            # edges = batch_data[i]["edges"]
            # theta_phis, taus = [], []
            # for j in range(conformer_positions.shape[0]):
            #     theta_phi, tau  = get_comenet_feature(conformer_positions[j], conformer_positions.shape[1], edges)
            #     theta_phis.append(theta_phi)
            #     taus.append(tau)
            # batch_data[i]["comenet_theta_phis"] = np.stack(theta_phis)
            # batch_data[i]["comenet_taus"] = np.stack(taus)


        df_3 = datamol.descriptors.batch_compute_many_descriptors(new_mol_batch,
                                                        properties_fn=descriptor_3d_fn_map,
                                                        add_properties=False,
                                                        batch_size=len(new_mol_batch),
                                                        n_jobs=self.num_threads)
        cols = df_3.columns
        for i in range(len(batch_data)):
            mol_descriptor = df_3.iloc[i]
            for col in cols:
                batch_data[i][col] = mol_descriptor[col]

        return batch_data

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
            "atomic_num": safe_index(Featurizer.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()) + 1,
            "chiral_tag": safe_index(Featurizer.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()) + 1,
            "degree": safe_index(Featurizer.atom_vocab_dict["degree"], atom.GetTotalDegree()) + 1,
            "explicit_valence": safe_index(Featurizer.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()) + 1,
            "formal_charge": safe_index(Featurizer.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()) + 1,
            "hybridization": safe_index(Featurizer.atom_vocab_dict["hybridization"], atom.GetHybridization()) + 1,
            "implicit_valence": safe_index(Featurizer.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()) + 1,
            "is_aromatic": safe_index(Featurizer.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())) + 1,
            "total_numHs": safe_index(Featurizer.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()) + 1,
            'num_radical_e': safe_index(Featurizer.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()) + 1,
            'atom_is_in_ring': safe_index(Featurizer.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())) + 1,
            'valence_out_shell': safe_index(Featurizer.atom_vocab_dict['valence_out_shell'],
                                            Featurizer.period_table.GetNOuterElecs(atom.GetAtomicNum())) + 1,
            'van_der_waals_radis': Featurizer.period_table.GetRvdw(atom.GetAtomicNum()),
            # 'partial_charge': Featurizer.check_partial_charge(atom),
            'mass': atom.GetMass() * 0.01,
             'in_num_ring_with_size3': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size3'], ring_list[0]) + 1,
            'in_num_ring_with_size4': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size4'], ring_list[1]) + 1,
            'in_num_ring_with_size5': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size5'], ring_list[2]) + 1,
            'in_num_ring_with_size6': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size6'], ring_list[3]) + 1,
            'in_num_ring_with_size7': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size7'], ring_list[4]) + 1,
            'in_num_ring_with_size8': safe_index(Featurizer.atom_vocab_dict['in_num_ring_with_size8'], ring_list[5]) + 1,
        }
        return feat

    def atom_to_feature_vector(self, atom, ring_list: Optional[List] = None):
        feature_names = self.atom_features
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
    def dihedral_check(mol : Chem.rdchem.Mol) -> bool:
        """check for dihedral pattern"""
        struct = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
        n_atoms = mol.GetNumAtoms()
        if n_atoms < 4:
            return False
        elif mol.GetNumBonds() < 4:
            return False
        elif not mol.HasSubstructMatch(struct):
            return False
        return True

    @staticmethod
    def potentially_reacted(mol : Chem.rdchem.Mol) -> bool:
        """mol had potential reaction"""
        try:
            canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception as e:
            return False
        return True

    @staticmethod
    def fragment_check(smiles: str) -> bool:
        """check if there are fragments in `smiles` representation"""
        return '.' in smiles

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_atom_poses(mol: Chem.rdchem.Mol, conf):
        """extract 3d pose"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def remove_hs(mol: Chem.rdchem.Mol,
                  implicit_only: bool = False,
                  update_explicit_count: bool = False,
                  sanitize: bool = True):
        """Removes hydrogens from a molecule.
        Args:
            mol: a molecule.
            implicit_only: whether to only remove implicit hydrogens.
            update_explicit_count: whether to update the explicit hydrogen count.
            sanitize: whether to sanitize the molecule after the hydrogens are removed.
        """
        mol = AllChem.RemoveHs(
            mol,
            implicitOnly=implicit_only,
            updateExplicitCount=update_explicit_count,
            sanitize=sanitize,
        )
        return mol

    @staticmethod
    def add_hs(mol: Chem.rdchem.Mol,
               explicit_only: bool = False,
               add_coords: bool = False,
               only_on_atoms: Optional[List[int]] = None,
               add_residue_info: bool = False):
        """Adds hydrogens to the molecule.
            Args:
                mol: a molecule.
                explicit_only: whether to only add explicit hydrogens.
                add_coords: whether to add 3D coordinates to the hydrogens.
                only_on_atoms: a list of atoms to add hydrogens only on.
                add_residue_info: whether to add residue information to the hydrogens.
                    Useful for PDB files.
        """
        mol = AllChem.AddHs(
            mol,
            explicitOnly=explicit_only,
            addCoords=add_coords,
            onlyOnAtoms=only_on_atoms,
            addResidueInfo=add_residue_info,
        )

        return mol

    @staticmethod
    def calculate_energy(mol: Chem.rdchem.Mol, conf_id : int, num_iters : int = 0, add_hs: bool = False) -> float:
        """ calculate `mol` force field energy.
            Args:
                mol (Chem.rdchem.Mol): Mol object
                conf_id (int): conformer id for FF calculation
                num_iters (int): number of iterations to minimize FF calculation (default: 0)
                add_hs (bool): add hydrogen to mol, will be removed prior to returning from fn
            Returns:
                Energy
        """
        if add_hs:
            mol = Featurizer.add_hs(mol)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        force_field = AllChem.MMFFGetMoleculeForceField(mol,
                                                        props,
                                                        conf_id)
        force_field.Initialize()
        energy = force_field.CalcEnergy()

        if num_iters > 0:
            energy = ff.Minimize(maxIts=num_iters)
        if add_hs:
            mol = Featurizer.remove_hs(mol)
        return energy

    def gen_conformers_rdkit(mol : Chem.rdchem.Mol,
                            add_hs: bool = False,
                            num_confs: int = 100,
                            max_attempts: int = 1000,
                            prune_rms_thresh: float = 0.1,
                            use_exp_torsion_angle_prefs: bool = True,
                            use_basic_knowledge: bool = True,
                            enforce_chirality: bool = True,
                            num_threads : int = 0):
        if add_hs:
            mol = Chem.AddHs(mol)

        ids = AllChem.EmbedMultipleConfs(mol,
                                            numConfs=num_confs,
                                            maxAttempts=max_attempts,
                                            pruneRmsThresh=prune_rms_thresh,
                                            useExpTorsionAnglePrefs=use_exp_torsion_angle_prefs,
                                            useBasicKnowledge=use_basic_knowledge,
                                            enforceChirality=enforce_chirality,
                                            numThreads=num_threads)

        if add_hs:
            mol = Chem.RemoveHs(mol)
        return list(ids)


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
    def list_mol_descriptors(filter: str = "") -> list:
        """ Returns the list of available descriptors

            Args:
                filter (str): can be one of ['2d', 'rdmol', 'datamol', '3d', 'crippen', 'lipinski', '']. Default behavior is to list all
                (default: '')

            Returns
                List
        """
        if filter == "rdmol":
            return list(_rdmol_descriptors.keys())
        elif filter == "2d":
            return list(_descriptors.keys())
        elif filter == "datamol":
            return list(_data_mol.key()) + list(_data_mol_descriptors.keys())
        elif filter == "3d":
            return list(_descriptors_3d.keys())
        elif filter == "crippen":
            return list(_crippen.keys())
        elif filter == "lipinski":
            return list(_lipinski.keys())
        elif filter == "kpgt":
            return list()
        else:
            return list(all_descriptors.keys())

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
