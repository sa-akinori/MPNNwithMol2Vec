# -*- coding: utf-8 -*-
#%%
import torch
import numpy as np
import pandas as pd
from torch import nn
from mol2vec import Mol2Vec
from makemol2vec import *
from model import MolGraph, MPN, RandomSeed
from sklearn.model_selection import train_test_split

class downstream:
    """
    This is an example of how to use mpnn and how to create graph.
    """
    def __init__(self, arg):

        self.arg = arg
        self.mpn = MPN(arg)
        self.mlp = nn.Sequential(*[nn.Linear(arg["hidden_size"], arg["hidden_size"]), nn.ReLU(), nn.Linear(arg["hidden_size"], 1)])
        self.criterion = nn.MSELoss()

        self.mpn, self.mlp = self.mpn.to(arg["device"]), self.mlp.to(arg["device"])

        param = list(self.mpn.parameters()) + list(self.mlp.parameters())
        self.optimizer = torch.optim.Adam(param, lr=0.0001)
        self.scaler = torch.cuda.amp.GradScaler()

    def get_graph(self, smiles, mol2vec):
        #Generating graph is time-consuming, so generate them in advance.
        self.graph = dict()
        for cpd in smiles:
            self.graph[cpd] = MolGraph(cpd, mol2vec)

    def predict(self, data):

        tensor, scope = self.mpn([self.graph[cpd] for cpd in data["nonstereo_aromatic_smiles"]])
        hiddens = list()
         
        for start, size in scope:
            
            hidden = tensor.narrow(0, start, size)
            hiddens.append(torch.sum(hidden, axis=0, keepdim=True))
            
        output = torch.cat(hiddens, axis=0)
        output = self.mlp(output)

        return output

    def train(self, tr_pd):
        
        self.set_train_eval("train")
        for epoch in range(self.arg["epoch"]):
            
            tr_pd = tr_pd.sample(frac=1, random_state=epoch).reset_index(drop=True)
            batches = [tr_pd.loc[i:i+self.arg["batch_size"]-1, :] for i in range(0, tr_pd.shape[0], self.arg["batch_size"])]
            
            for batch in batches:
                
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    predict = self.predict(batch)
                    observe = torch.FloatTensor(np.array(batch["measured log solubility in mols per litre"]).reshape(-1, 1)).to(self.arg["device"])
                    loss = self.criterion(predict, observe)
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
               
    def test(self, te_pd):
        
        self.set_train_eval("eval")
        pre = self.predict(te_pd)
        
        return pre.to("cpu").detach().numpy().copy()
        
    def set_train_eval(self, mode):
        
        if mode == "train":
            self.mpn.train()
            self.mlp.train()
            
        elif mode == "eval":
            self.mpn.eval()
            self.mlp.eval()
        
        
#%%
if __name__=="__main__":
    folder = "/home/sato_akinori/MPNNwithMol2Vec/data/" #Please change this part to fit on your work.
    radius = 3
    
    if 1:
        #Train mol2vec model
        #folder shows the absolute path where save data of compounds for mol2vec.
        make_corpus(base_folder=folder, file_name="example_mol2vec", radius=radius)
        train_mol2vec(base_folder=folder,radius=radius)
    
    if 1:
        #Generate mol2vec with smiles of downstream task
        mol2vec_model = folder + "mol2vec/mol2vec_radius%s.bin"%radius 
        task = pd.read_csv("../data/example_downstream.csv", index_col=0)
        cpds = list(set([smiles for smiles in task["nonstereo_aromatic_smiles"]]))
        cpds = pd.Series(cpds)
        vals = Mol2Vec.GetRepresentation(cpds, mol2vec_model, radius=3, return_atomwise=True)
        vals.to_pickle(folder+"mol2vec/mol2vec.pkl", protocol=0)

    if 1:
        RandomSeed(0)
        #train and test of downstream task(Example of demonstration)
        task = pd.read_csv("../data/example_downstream.csv", index_col=0).drop_duplicates(subset="nonstereo_aromatic_smiles", keep=False)
        train, test = train_test_split(task, test_size=0.1)
        arg = {"device":"cuda", "epoch":2, "depth":4, "dr_ratio_mpn":0.0, "hidden_size":400, 
               "agn_num":2, "dr_mpn":True, "bias_mpn":True, "batch_size":16} #depth, dr_ratio_mpn, hidden_size, agn_num  are hyperparameters.
        mol2vec = pd.read_pickle(folder+"mol2vec/mol2vec.pkl")
        model = downstream(arg)
        cpds  = [cpd for cpd in task["nonstereo_aromatic_smiles"]]
        model.get_graph(cpds, mol2vec)
        model.train(train.reset_index(drop=True))
        predict = model.test(test.reset_index(drop=True))
        print(predict)
