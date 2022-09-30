import rdkit
from rdkit import Chem
from rdkit.Chem import rdDistGeom, AllChem, rdMolAlign


def add_conformers(mol, num_conformers):
    if num_conformers == 0:
        return mol
    param = rdDistGeom.ETKDGv2()
    param.pruneRmsThresh = 0.1  # prunermsthresh'
    numconf = 50
    cids = rdDistGeom.EmbedMultipleConfs(mol, numconf, param)
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    _ = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')
    res = []

    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
        e = ff.CalcEnergy()
        res.append((cid, e))
    sorted_res = sorted(res, key=lambda x: x[1])
    rdMolAlign.AlignMolConformers(mol)
    for cid, e in sorted_res:
        mol.SetProp('CID', str(cid))
        mol.SetProp('Energy', str(e))

    return mol
