from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from collections import defaultdict
import numpy as np

# global variable
rdkit_descriptorlist = [desc[0] for desc in Descriptors._descList]

def OrganizeMorganHashOnAtoms(mol: Chem.Mol, bitinf: dict):
	ret_dict = defaultdict(str)

	for atom in mol.GetAtoms():
		idx = atom.GetIdx()
		hash_radi = []
		for hash, val in bitinf.items():
			ok_hash = [(hash, v[1]) for v in val if v[0] == idx]
			hash_radi += ok_hash
		ordered = sorted(hash_radi, key=lambda x: x[1])
		ret_dict[idx] = [v[0] for v in ordered] # only select hash value
	return ret_dict
	

def HeavyAtomCount(smi: str):
	mol = Chem.MolFromSmiles(smi)
	if mol is None:
		print(f'cannot conver smiles {smi} to Mol. Return None')
		return None
	
	return mol.GetNumAtoms()

def GetMorganFeatures(orgmol, radius=2, return_as_string=False, input_smiles=True, return_atom_hash=False, include_duplicate_hashes=False):
	if input_smiles:
		try:
			mol = Chem.MolFromSmiles(orgmol)
		except:
			print("conversion error calculation ecfp skip.")
			print(orgmol)
			return None
	else:
		mol = orgmol

	if mol is None:
		print('Molecule is None. Return empty data')
		print(orgmol)
		return None
	binf = dict()
	features = AllChem.GetMorganFingerprint(mol, radius=radius, bitInfo=binf, includeRedundantEnvironments=include_duplicate_hashes)
	finf = features.GetNonzeroElements()

	# return options
	if return_atom_hash:
		atomidx_hash = OrganizeMorganHashOnAtoms(mol, binf)
		return atomidx_hash

	if return_as_string:
		return str(finf)
	else:
		return finf


def MorganFeature2BitVector(feature_dict, nbits=2048, return_idx=False):
	onbits = [n%nbits for n in feature_dict.keys()]
	if return_idx:
		return onbits
	v = np.zeros(nbits, dtype=bool)
	v[onbits] = True
	return v

def CalcECFPFeatureLevels(mol, radius=2):
    """
    atom index is the key of the feature level
    """
    ecfp = GetMorganFeatures(mol, radius, input_smiles=False, return_atom_hash=True, include_duplicate_hashes=True)
    
    return ecfp # feature level list is return.
