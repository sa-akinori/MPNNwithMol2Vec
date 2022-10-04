import pandas as pd
import numpy as np
import torch

##まずmakemol2vec.pyによりmol2vecモデルを作成してください。
mol2vec_model = "/home/sato/Daikin-RA/DataFrame/Mol2Vec_suzuki/data/mol2vec_radius3/mol2vec_hash_radius3_chembl_minCount1.bin"
data = pd.read_csv("/home/sato/Daikin-RA/DataFrame/reaction_data/suzuki-miyaura/suzuki-miyaura.csv", index_col=0)
targets, names = ["Reactant_1_SMILES", "Reactant_2_SMILES", "Solvent_SMILES", "Reagent_SMILES", "Ligand_SMILES"], ["Reactant_1", "Reactant_2", "Solvent", "Reagent", "Ligand"]

all_vals = pd.DataFrame()
for target, name in zip(targets, names):
    dup_data = list(set([smiles for smiles in data[target] if smiles != "None"]))
    dup_data = list(set(list(itertools.chain.from_iterable([smiles.split(".") for smiles in dup_data]))))
    dup_data = pd.Series(dup_data)
    vals = Mol2Vec.GetRepresentation(dup_data, mol2vec_model, radius=3, return_atomwise=True)
    vals.to_pickle("DataFrame/reaction_data/suzuki-miyaura/mol2vec_data/Mol2Vec_%s_r3.pkl"%name, protocol=0)
    all_vals = pd.concat([all_vals, vals], axis=0)

all_vals = all_vals.drop_duplicates(subset="input_washed_smiles")
all_vals.to_pickle("DataFrame/reaction_data/suzuki-miyaura/mol2vec_data/Mol2Vec_suzuki_r3.pkl", protocol=0)