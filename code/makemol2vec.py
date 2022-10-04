# -*- coding: utf-8 -*-
#%%
import pandas as pd
from mol2vec import Mol2Vec

def make_corpus(base_folder, target_name, radius=2, use_input_smiles=True):

    if use_input_smiles:
        smiles = '%s/input_smiles.txt'%base_folder
        curated_smiles = pd.read_csv(smiles, sep="\t", index_col=0)
        Mol2Vec.MakeFastTextInputFile(curated_smiles["nonstereo_aromatic_smiles"], folder_name=base_folder + '/mol2vec', radius=radius, use_hash=True, njobs=10, debug=False, use_curated_smiles=True)
    else:
        chembl_mols = pd.read_csv('%s/compounds/%s.txt'%(base_folder, target_name), sep='\t', index_col=0)
        Mol2Vec.MakeFastTextInputFile(chembl_mols['nonstereo_aromatic_smiles'], folder_name='%s/mol2vec'%base_folder, radius=radius, debug=False)

def train_mol2vec(base_folder):

    Mol2Vec().RunFastText('%s/mol2vec/hash_sentence.txt'%base_folder,
                            '%s/mol2vec/mol2vec_hash_radius2.bin'%base_folder, epoch=30, ws=15, minCount=1)

#%%
if __name__ == '__main__':
    make_corpus(base_folder=folder, target_name="compounds", radius=2, use_input_smiles=False)
    train_mol2vec(base_folder=folder)