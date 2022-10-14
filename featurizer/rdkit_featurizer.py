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
    return list(map(len, [RDKITFeaturizer.atom_vocab_dict[name] for name in list_of_acquired_feature_names]))

def get_bond_feature_dims(list_of_acquired_feature_names):
    list_bond_feat_dim = list(map(len, [RDKITFeaturizer.bond_vocab_dict[name] for name in list_of_acquired_feature_names]))
    # +1 for self loop edges
    return [_l + 1 for _l in list_bond_feat_dim]

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
                 num_workers: int = 0,
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
        self.num_workers = num_workers
        self.include_rings = any(['in_num_ring_with' in af for af in self.atom_features])
        self.add_comenet_features = add_comenet_features

    def smiles_to_graph(self,
                        smiles: Union[List, str],
                        mol_3d: Optional[Union[List[Chem.rdchem.Mol], Chem.rdchem.Mol]]=None):
        if not isinstance(smiles, list):
            smiles = [smiles]
        if mol_3d is not None:
            if not isinstance(mol_3d, list):
                mol_3d = [mol_3d]
        mol_batch = []
        for i, smile in enumerate(smiles):
            if self.skip_fragment_smiles and Featurizer.fragment_check(smile):
                continue
            mol = Featurizer.smiles_to_mol(smile, package="rdkit")
            if len(mol.GetAtoms()) == 0:
                continue
            mol_batch.append(mol)

        if not len(mol_batch):
            return None
        batch_data = []
        for mol in mol_batch:
            if self.skip_dihedral and Featurizer.dihedral_check(mol):
                continue
            data = defaultdict(list)

            # - atom features
            ring_list = None
            if self.include_rings:
                ring_list = Featurizer.get_ring_size(mol)
            for i, atom in enumerate(mol.GetAtoms()):
                feat_dict = self.atom_to_feature_vector(atom, ring_list[i])
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
                        if bond_feature is not None:
                            data[bond_feat_name] += [bond_feature] * 2
            else:   # mol has no bonds
                for bond_feat_name in self.bond_features:
                    data[bond_feat_name] = np.zeros((0, ), dtype=np.int64)
                data["edges"] = np.zeros((0, 2), dtype=np.int64)
            data["dist_matrix"] = rdkit.Chem.rdmolops.GetDistanceMatrix(mol)
            if self.add_self_loop:
                for i in range(N):
                    data["edges"] += [(i, i)]
                for bond_feat_name in self.edge_features:
                    bond_id = get_bond_feature_dims(bond_feat_name)[0] - 1
                    data[bond_feat_name] += [bond_id] * N

            data["edges"] = np.array(data["edges"], np.int64)
            # - comenet features
            if self.add_comenet_features and mol_3d is not None:
                data["pos"] = mol_3d[i].GetPositions()
                data["comenet_dist_theta_phi"], data["comenet_dist_tau"] = get_comenet_feature(data)

            batch_data.append(data)
        # - descriptors
        if self.descriptor_features_map:
            df = datamol.descriptors.batch_compute_many_descriptors(batch_mols,
                                                                    properties_fn=self.descriptor_features_map,
                                                                    add_properties=False,
                                                                    batch_size=len(batch_mols),
                                                                    n_jobs=self.num_workers)
            cols = df.columns
            for i in range(len(batch_data)):
                mol_descriptor = df.iloc[i]
                for col in cols:
                    batch_data[i][col] = mol_descriptor[col]

        return batch_data

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
        try:
            return safe_index(Featurizer.bond_vocab_dict[name], Featurizer.get_bond_value(bond, name))
        except:
            return None

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
    def gen_conformers(mol : Chem.rdchem.Mol, package="rdkit", **kwargs) -> Chem.rdchem.Mol:
        if package == "rdkit":
            ids = gen_conformers_rdkit(mol, **kwargs)
        elif package == "datamol": # - by default will generate all conformers
            # - https://doc.datamol.io/stable/api/datamol.conformers.html#datamol.conformers._conformers.generate
            mol = datamol.conformers.generate(mol, **kwargs)
        return mol

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            energy = res[index][1]
            conf = new_mol.GetConformer(id=int(index))
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Featurizer.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_atom_poses(mol, conf):
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_atom_feature_id(atom, name):
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

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
            return int(atom.GetMass())
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
