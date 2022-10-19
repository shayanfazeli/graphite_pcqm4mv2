import json
from typing import Union, List, Optional
from pathlib import Path

import torch

from featurizer.rdkit_featurizer import Featurizer
from featurizer.rdkit_featurizer import ogb_atom_features, ogb_bond_features
from featurizer.rdkit_featurizer import get_atom_feature_dims, get_bond_feature_dims

class DataConnector():
    def __init__(self, 
                 featurizer,
                 atom_features: Optional[List[str], str] = "ogb",
                 edge_features: Optional[List[str], str] = "ogb",
                 descriptors_2d: Optional[List[str], str] = "ogb",
                 descriptors_3d: Optional[List[str], str] = "ogb",
                 conformers: bool = False,
                 include_ring_sizes: bool = True,
                 return_one_large_embedding: bool = False,
                 embed_descriptors: bool = False,
                 include_finger_prints: bool = False,
                 include_daylight_fg_counts: bool = False,
                 ):
        """
            XXX: THIS IS INCOMPLETE... 
            
            The purpose of this object is to provude fn's:
                - filter out features from the list of all extracted features and returning only desired features
                - option to convert int columns features to single large embedding value 


            featurizer : configured featurizer used to generate the data
            atom_features: list of atom features to include in `Data`
            edge_features: list of edge features to include in `Data`
            descriptors_2d: list of atom features to include in `Data`
            descriptors_3d: list of edge features to include in `Data`
            conformers: whether to include conformers or not
        """

        self.data_path = Path(data_path)

        self.metadata_2d = featurizer.get_descriptors2d_metadata()
        self.metadata_3d = featurizer.get_descriptors3d_metadata()

        self.atom_feature_order = featurizer.get_atom_feature()
        self.bond_feature_order = featurizer.get_bond_features()
        self.descriptor_2d_order = list(self.metadata_2d.keys())
        self.descriptor_3d_order = list(self.metadata_3d.keys())

        self.bond_features = self.get_bond_features()
        self.atom_features = self.get_atom_features()

        self.descriptors_2d = self.get_descriptor2d_features()
        self.descriptors_3d = self.get_descriptor3d_features()
        self.conformers = conformers
        self.include_ring_sizes = include_ring_sizes
        self.include_finger_prints = include_finger_prints
        self.embed_descriptors = embed_descriptors

    def process(self, data: Data) -> Data:
        if not self.include_ring_sizes:
            data.remove_tensor("ring_sizes")
        if not self.conformers:
            data.remove_tensor("conformer_pos")
        if self.descriptors_2d is None:
            data.remove_tensor("descriptor_2d")
        
        if not self.include_finger_prints:
            data.remove_tensor("fingerprint") 
        
        if not self.include_daylight_fg_counts:
            data.remove_tensor("fingerprint") 

    def filter_atom_features(self, data):

    def get_atom_features(self, atom_feature):     
        if isinstance(atom_features, list):
            atom_features = atom_features
        elif atom_features == "ogb":
            atom_features = ogb_atom_features
        elif atom_features == "all":
            atom_features = featurizer.get_atom_features()
        else:
            raise ValueError(f"undefined {atom_feature}")
        return atom_features

    def get_bond_features(self, bond_features):     
        if isinstance(bond_features, list):
            bond_features = bond_features
        elif bond_features == "ogb":
            bond_features = ogb_bond_features
        elif bond_features == "all":
            bond_features = featurizer.get_bond_features()
        else:
            raise ValueError(f"undefined {atom_feature}")
        return bond_features

    def get_descriptor2d_features(self, descriptor2d_features):     
        if isinstance(descriptor2d_featuress, list):
            descriptor2d_featuress = descriptor2d_featuress
        elif descriptor2d_featuress == "ogb":
            descriptor2d_featuress = None
        elif descriptor2d_featuress == "all":
            descriptor2d_featuress = featurizer.get_descriptors2d_features()
        else:
            raise ValueError(f"undefined {atom_feature}")
        return descriptor2d_features

    def get_descriptor3d_features(self, descriptor2d_features):     
        if isinstance(descriptor2d_featuress, list):
            descriptor2d_featuress = descriptor2d_featuress
        elif descriptor2d_featuress == "ogb":
            descriptor2d_featuress = None
        elif descriptor2d_featuress == "all":
            descriptor2d_featuress = featurizer.get_descriptors2d_features()
        else:
            raise ValueError(f"undefined {atom_feature}")
        return descriptor2d_features
    
    
    def init_large_embedding(self):
        atom_dims = get_atom_feature_dims(self.atom_features)
        bon_dims = get_bond_feature_dims(self.bond_features)
        if self.embed_descriptors:

        

    def process(self, data):

    def init_normalizing_vector(self, metadata):
        norm_vec = np.ones((1, len(metadata)))
        i = 0
        for k, v in metadata.items():
            if metadata[k]['dtype'] == "float64":
                norm_vec[i]
        