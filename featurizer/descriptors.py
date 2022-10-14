import re
from typing import Callable
from functools import partial

import rdkit
from rdkit import Chem
import datamol

from .kpgt_descriptors.rdDescriptors import FUNCS as KPGT_FUNCS
from .kpgt_descriptors.rdNormalizedDescriptors import cdfs
from .kpgt_descriptors.dists import dists

import logging

_kpgt_descriptors = {k:k for k in dists.keys()}

def _cap_str_cammel_case_str(s):
    i = 0
    l = len(s)
    p = []
    for j in range(1, l):
        if s[j].isupper() and j + 1 < l and s[j+1].islower() and s[j-1].islower():
            p.append(s[i:j].lower())
            i = j
    p.append(s[i:].lower())
    return '_'.join(p)

# lambda x: '_'.join(re.findall('([A-Z]*[^A-Z]*)', str(x))).lower()

# - https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
__rdmol_descriptors = [
    "CalcAUTOCORR2D",
    "CalcNumAmideBonds",
    "CalcNumAtoms",
    "CalcNumBridgeheadAtoms",
    "CalcNumHBA",
    "CalcNumHBD",
    "CalcNumHeavyAtoms",
    "CalcNumHeterocycles",
    "CalcNumLipinskiHBA",
    "CalcNumLipinskiHBD",
    "CalcNumRings",
    "CalcNumSpiroAtoms",
    "CalcPhi",
]
_rdmol_descriptors = {_cap_str_cammel_case_str(s): s for s in __rdmol_descriptors}

# - https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html
_descriptors = [
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "HeavyAtomMolWt",
    "MaxAbsPartialCharge",
    "MaxPartialCharge",
    "MinAbsPartialCharge",
    "MinPartialCharge",
    "MolWt",
    "NumRadicalElectrons",
    "NumValenceElectrons",
]
_descriptors = {_cap_str_cammel_case_str(s): s for s in _descriptors}

# - https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors3D.html
__descriptors_3d = [
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "NPR1",
    "NPR2",
    "PMI1",
    "PMI2",
    "PMI3",
    "RadiusOfGyration",
    "SpherocityIndex",
     # - below are from https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
    "CalcAUTOCORR3D",
    "CalcRDF",
    "CalcPBF",
]
_descriptors_3d = {_cap_str_cammel_case_str(s): s for s in __descriptors_3d}

__data_mol = [
    "n_aromatic_atoms_proportion",
    "n_charged_atoms",
    "n_rigid_bonds",
    "n_stereo_centers"
]
_data_mol = {s: s for s in __data_mol}

__data_mol_descriptors = [
    "n_aromatic_atoms",
]
_data_mol_descriptors = {s: s for s in __data_mol_descriptors}

__lipinski = [
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
]
_lipinski = {_cap_str_cammel_case_str(s): s for s in __lipinski}

_crippen = [
    "MolMR" 
]
_crippen = {_cap_str_cammel_case_str(s): s for s in _crippen}



all_descriptors = {
                   **_descriptors_3d, 
                #    **_descriptors, # - covered by _kpgt_descriptors
                   **_rdmol_descriptors,
                   **_data_mol,
                   **_data_mol_descriptors,
                #    **_lipinski, # - covered by _kpgt_descriptors
                #    **_crippen, # - covered by _kpgt_descriptors
                   **_kpgt_descriptors,
                  }

descriptors_2d = {
    **_kpgt_descriptors,
    **_rdmol_descriptors,
    **_data_mol,
    **_data_mol_descriptors,
}

descriptors_3d = {
    **_descriptors_3d,
}
class FuncWrapper(object):
    def __init__(self, method_name, module_name):
        self.method_name = method_name
        self.module_name = module_name
    
    def __call__(self, mol):
        method = getattr(__import__(self.module_name, globals(), locals(), [self.method_name, ]), 
                         self.method_name)
        try: 
            return method(mol)
        except Exception as e:
            logging.exception("function application failed (%s)", e)

            return None
    
def fetch_descriptor_fn(name: str) -> Callable:
    """Return a descriptor function by name either from
    `Chem.Descriptors` or `Chem.rdMolDescriptors` or `Chem.Descriptors3D` or `datamol`
    Args:
        name: Descriptor name.
    """
    module_name = ""
    method_name = ""

    if name in all_descriptors:
        name = all_descriptors[name]

    fn = getattr(Chem.Descriptors, name, None)
    if fn is None: # - kpgt
        fn = getattr(KPGT_FUNCS, name, None)

    if fn is None: #
        fn = getattr(Chem.rdMolDescriptors, name, None)

    if fn is None: # - 3D
        fn = getattr(Chem.Descriptors3D, name, None)

    if fn is None: # - datamol
        fn = getattr(datamol, name, None)
    
    if fn is None: # - datamol descriptors
        fn = getattr(datamol.descriptors, name, None)

    if fn is None: # - Lipinski
        fn = getattr(Chem.Lipinski, name, None)

    if fn is None: # - Crippen
        fn = getattr(Chem.Crippen, name, None)

    if fn is not None:
        module_name = fn.__module__
        method_name = name
        fn = FuncWrapper(method_name, module_name)
    
    if fn is None:
        raise ValueError(f"Descriptor {name} not found.")

    return fn

descriptor_fn_map = {k: fetch_descriptor_fn(k) for k in all_descriptors}
descriptor_2d_fn_map = {k: fetch_descriptor_fn(k) for k in descriptors_2d}
descriptor_3d_fn_map = {k: fetch_descriptor_fn(k) for k in descriptors_3d}
