# -*- coding: utf-8 -*-
#%%
import pandas as pd
from mol2vec import Mol2Vec

def make_corpus(base_folder, file_name, radius=2):

    curated_smiles = pd.read_csv('%s/%s.txt'%(base_folder, file_name), sep="\t", index_col=0)
    Mol2Vec.MakeFastTextInputFile(curated_smiles["nonstereo_aromatic_smiles"], folder_name=base_folder + '/mol2vec', radius=radius, njobs=10)


def train_mol2vec(base_folder, radius):

    Mol2Vec().RunFastText('%s/mol2vec/hash_sentence.txt'%base_folder,
                            '%s/mol2vec/mol2vec_radius%s.bin'%(base_folder, radius), epoch=30, ws=15, minCount=1)

    
