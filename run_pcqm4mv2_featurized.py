from featurizer.rdkit_featurizer import Featurizer
from pyg_pcqm4mv2_featurized import PygPCQM4Mv2FeaturizedDataset



if __name__ == "__main__":
    
    featurizer = Featurizer(
        atom_features=["atomic_num", "chiral_tag", "degree", "formal_charge", "total_numHs", 'num_radical_e', "hybridization", 
                    "is_aromatic", 'atom_is_in_ring', "explicit_valence", "implicit_valence", 'valence_out_shell', "van_der_waals_radis", 'mass'],
        bond_features=["bond_dir", "bond_type", "is_in_ring", 'dist_matrix', 'bond_stereo', 'is_conjugated'],
        add_self_loop=True,
        include_rings=True,
        generate_conformer=True,
        num_conformers=1,
        generate_descriptor2d=True,
        generate_descriptor3d=True,
        generate_additional_conformers_for_sdf=False,
    )

    dataset = PygPCQM4Mv2FeaturizedDataset(featurizer, 
                                        root='/workspace/data/',
                                        include_sdf=True,
                                        num_workers=32)