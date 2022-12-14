# -*- coding: utf-8 -*-
#%%
import torch
import random
import numpy as np
import torch.nn as nn
from rdkit import Chem

ATOM_FDIM, BOND_FDIM = 400, 6

def RandomSeed(seed):
    """
    It is necessary to make deterministic work.
    """
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True

def Mol2Vec(smiles, idx, reference):
    mol2vec = reference.query("input_smiles == @smiles").reset_index(drop=True)
    return np.hstack(mol2vec.loc[0, idx]).tolist()

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def BondGenerate(bond):
    if bond.GetIsAromatic():
        bt = 0 # the same as unspecified in RDKit
    else:
        bt = int(bond.GetBondType())
    fbond = [bond.IsInRing()] + onek_encoding_unk(bt, [0, 1, 2, 3, 4])
    return fbond

def index_select_ND(source, index):

    source = source.float()
    index = index.long()
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source[index.view(-1), :]
    ##torch.use_deterministic_algorithms doesn't work if we use the below code.
    # target = source.index_select(dim=0, index=index.view(-1))
    target = target.view(final_size)

    return target

class MolGraph:
    """
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2a: A mapping from an atom index to the index of the neighbor atoms.
    - a2b: A mapping from an atom index to a list of bond indices.
    """

    def __init__(self, smiles, mol2vec):

        if smiles == "None":
            self.n_atoms, self.n_bonds = 0, 0
            self.f_atoms, self.f_bonds = [], []
            self.a2a, self.a2b = [], []

        else:
            mol = Chem.MolFromSmiles(smiles)
            self.n_atoms, self.n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()
            self.f_atoms, self.f_bonds = [[] for _ in range(self.n_atoms)], [[] for _ in range(self.n_bonds)]
            self.a2a, self.a2b = [[] for _ in range(self.n_atoms)], [[] for _ in range(self.n_atoms)]
            
            #mol2vec of each molecule is saved in mol2vec file, so we can't extract feature of concatenated smiles by dot like "a.b".
            #Therefore, we split smiles.
            mid_idx = 0
            sep_smiles = smiles.split(".")
            for s_smile in sep_smiles:
                for s_atom in Chem.MolFromSmiles(s_smile).GetAtoms():
                    idx = s_atom.GetIdx()
                    f_atoms = Mol2Vec(s_smile, idx, mol2vec)
                    self.f_atoms[idx + mid_idx] = f_atoms
                   
                mid_idx += idx+1

            for atom1 in mol.GetAtoms():

                neigh = [a for a in atom1.GetNeighbors()]
                self.a2a[atom1.GetIdx()] = [a.GetIdx() for a in neigh]

                bonds = [mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx()) for atom2 in neigh]
                bond_idx = [bond.GetIdx() for bond in bonds]
                self.a2b[atom1.GetIdx()] = bond_idx

                for b_id, bond in zip(bond_idx, bonds):

                    f_bond = BondGenerate(bond)
                    self.f_bonds[b_id] = f_bond

class BatchMolGraph:
    """
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs, args):

        self.args = args
        self.n_atoms = 1
        self.n_bonds = 1
        f_atoms = [[0] * ATOM_FDIM]
        f_bonds = [[0] * BOND_FDIM]
        self.a_scope = []

        a2a, a2b = [[]], [[]]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            a2a.extend([[b2 + self.n_atoms for b2 in b1] for b1 in mol_graph.a2a])
            a2b.extend([[b2 + self.n_bonds for b2 in b1] for b1 in mol_graph.a2b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        max_num_bonds = max(len(in_bonds) for in_bonds in a2b)
        max_num_atoms = max(len(nei_atoms) for nei_atoms in a2a)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.a2a = torch.LongTensor([a2a[a] + [0] * (max_num_atoms - len(a2a[a])) for a in range(self.n_atoms)])

    def get_components(self):

        return self.f_atoms, self.f_bonds, self.a2a, self.a2b, self.a_scope


class MPNEncoder(nn.Module):

    def __init__(self, args):

        super(MPNEncoder,self).__init__()

        self.args  = args

        if args["dr_mpn"]:
            self.W_i  = nn.Linear(ATOM_FDIM, args["hidden_size"], bias=args["bias_mpn"])
            self.tune = nn.PReLU()
            self.W_o  = nn.Sequential(*[nn.Linear(args["hidden_size"]*2, args["hidden_size"], bias=args["bias_mpn"]), nn.PReLU()])
            selfmodule = [nn.Linear(ATOM_FDIM, args["hidden_size"], bias=args["bias_mpn"]), nn.PReLU()]

            for i in range(args["depth"]-1):
                selfmodule.extend([nn.Dropout(p=args["dr_ratio_mpn"]), nn.Linear(args["hidden_size"], args["hidden_size"], bias=args["bias_mpn"]), nn.PReLU()])

        elif args["norm_type"] == "BatchNorm":

            self.W_i  = nn.Linear(ATOM_FDIM, args["hidden_size"], bias=args["bias_mpn"])
            self.tune = nn.Sequential(*[nn.BatchNorm1d(args["hidden_size"]), nn.PReLU()])
            self.W_o  = nn.Sequential(*[nn.Linear(args["hidden_size"]*2, args["hidden_size"], bias=args["bias_mpn"]), nn.BatchNorm1d(args["hidden_size"]), nn.PReLU()])
            selfmodule = [nn.Linear(ATOM_FDIM, args["hidden_size"], bias=args["bias_mpn"]), nn.BatchNorm1d(args["hidden_size"]), nn.PReLU()]

            for _ in range(args["depth"]-1):

                selfmodule.extend([nn.Linear(args["hidden_size"], args["hidden_size"], bias=args["bias_mpn"]), nn.BatchNorm1d(args["hidden_size"]), nn.PReLU()])

        elif args["norm_type"] == "LayerNorm":

            self.W_i  = nn.Linear(ATOM_FDIM, args["hidden_size"], bias=args["bias_mpn"])
            self.tune = nn.Sequential(*[nn.LayerNorm(args["hidden_size"]), nn.PReLU()])
            self.W_o  = nn.Sequential(*[nn.Linear(args["hidden_size"]*2, args["hidden_size"], bias=args["bias_mpn"]), nn.LayerNorm(args["hidden_size"]), nn.PReLU()])
            selfmodule = [nn.Linear(ATOM_FDIM, args["hidden_size"], bias=args["bias_mpn"]), nn.LayerNorm(args["hidden_size"]), nn.PReLU()]

            for _ in range(args["depth"]-1):

                selfmodule.extend([nn.Linear(args["hidden_size"], args["hidden_size"], bias=args["bias_mpn"]), nn.LayerNorm(args["hidden_size"]), nn.PReLU()])

        self.W_ah = nn.Sequential(*selfmodule)

        for i in range(args["depth"]):

            exec(f"modulelist{i} = self.makemodulelist()")
            exec(f"self.W_h{i} = nn.Sequential(*modulelist{i})")

    def makemodulelist(self):

        modulelist = [nn.Linear(self.args["hidden_size"] + BOND_FDIM, self.args["hidden_size"], bias=self.args["bias_mpn"])]

        if self.args["dr_mpn"]:

            for _ in range(self.args["agn_num"]-1):

                modulelist.extend([nn.PReLU(), nn.Dropout(p=self.args["dr_ratio_mpn"]), nn.Linear(self.args["hidden_size"], self.args["hidden_size"], bias=self.args["bias_mpn"])])

        elif self.args["norm_type"] == "BatchNorm":

            for _ in range(self.args["agn_num"]-1):

                modulelist.extend([nn.BatchNorm1d(self.args["hidden_size"]), nn.PReLU(), nn.Linear(self.args["hidden_size"], self.args["hidden_size"], bias=self.args["bias_mpn"])])

        elif self.args["norm_type"] == "LayerNorm":

            for _ in range(self.args["agn_num"]-1):

                modulelist.extend([nn.LayerNorm(self.args["hidden_size"]), nn.PReLU(), nn.Linear(self.args["hidden_size"], self.args["hidden_size"], bias=self.args["bias_mpn"])])

        return modulelist

    def forward(self, mol_graph):
        """
        Encodes a batch of molecular graphs.
        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        f_atoms, f_bonds, a2a, a2b, a_scope = mol_graph.get_components()
        f_atoms, f_bonds, a2b, a2a    = f_atoms.to(self.args["device"]), f_bonds.to(self.args["device"]), a2b.to(self.args["device"]), a2a.to(self.args["device"])
        self.W_i, self.W_o, self.W_ah = self.W_i.to(self.args["device"]), self.W_o.to(self.args["device"]), self.W_ah.to(self.args["device"])
        self.tune = self.tune.to(self.args["device"])

        for i in range(self.args["depth"]):
            exec(f"self.W_h{i} = self.W_h{i}.to(self.args['device'])")

        input = self.W_i(f_atoms)
        self_message = input
        message = self_message.clone()
        
        self_message[0, :], message[0, :] = 0, 0 #It is bad to exist value in row 0, because row 0 is used to adjust the number of neighborhood atoms.

        for depth in range(self.args["depth"]):

            nei_a_message = index_select_ND(message, a2a)
            nei_f_bonds   = index_select_ND(f_bonds, a2b)
            n_message = torch.cat([nei_a_message, nei_f_bonds], dim=2)
            message   = n_message.sum(dim=1)

            message = eval(f"self.W_h{depth}(message)")
            self_message = self_message + message
            message = self_message.clone()
            self_message[0, :], message[0, :] = 0, 0 

        nei_a_message = index_select_ND(message, a2a)
        a_message  = self.tune(nei_a_message.sum(dim=1))
        cc_message = self.W_ah(f_atoms)
        a_input = torch.cat([cc_message, a_message], dim=1)
        atom_hiddens = self.W_o(a_input)

        return atom_hiddens, a_scope

class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args):

        super(MPN, self).__init__()
        self.args      = args
        self.encoder   = MPNEncoder(self.args)

    def forward(self,
                batch) -> torch.FloatTensor:

        batch = BatchMolGraph(batch, self.args)
        output = self.encoder.forward(batch)

        return output
