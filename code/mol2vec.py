import pandas as pd
import numpy as np
import multiprocessing
import os
import fasttext
from rdkit import Chem
from fingerprints import CalcECFPFeatureLevels
from utility import MakeFolder, AssertTerminate
from collections import defaultdict, OrderedDict


class Mol2Vec():
    """
    Mol2Vec transformation of featuring a set of smiles

    1. Making latent space representation model
    2. Appling the model to each of molecule to get latent representations
    """

    def __init__(self):
        pass

    @staticmethod
    def MakeFastTextInputFile(pd_smiles, folder_name='mol2vec', radius=1, njobs=10):
        """
        input_smiles: Skipt the curation part for saving the time (or reuse already curated smiles file)
        """

        folder = MakeFolder(folder_name)
        pd_smiles = pd_smiles.drop_duplicates()

        # Make the Sentence for them
        input_worker = MakeInputWorker(folder, radius, pd_smiles, njobs)

        pool = multiprocessing.Pool(njobs)
        pool.map(GetFeatureSentences_worker, input_worker)
        pool.close()
        pool.join()

        # cat the hash files (remove the current files later) / append statistics
        flist_h = ['%s/hash_%d.txt'%(folder, id) for id in range(njobs)]
        GetStatsCatFiles(flist_h, '%s/hash_sentence.txt'%folder, remove_input=True)

    @staticmethod
    def RunFastText(infile, outfile, ws=10, dim=100, minn=0, maxn=0, neg=5, epoch=20, minCount=1):
        model = fasttext.train_unsupervised(infile, model='skipgram', ws=ws, dim=dim, minn=minn, maxn=maxn,
                                            neg=neg, epoch=epoch, minCount=minCount) # off n-grams

        model.save_model(outfile)

    @staticmethod
    def GetRepresentation(pd_smiles, model_file, radius=1, return_atomwise=False):
        """
        IF return_atomwise is false, all the vectors are summed to generated fixed-size matrix
        Otherwise the return contains the atom-wise information of mol2vec
        """

        AssertTerminate(isinstance(pd_smiles, pd.Series))
        model = fasttext.load_model(model_file)

        # get the setntence
        ret_db = OrderedDict()

        for key, smiles in pd_smiles.items():
            vals = GetRepForSmiles(smiles, radius=radius, model=model)
            if vals is None:
                print("Found unidentifed words: (%s\t%s)"%(key, smiles))
                continue
            if not return_atomwise:
                val_radius = defaultdict(int)
                for atom_key in vals:
                    for idx in range(radius+1):
                        val_radius[idx] += vals[atom_key][idx]

                # making lables for putting values into dataframe
                pd_dict = dict()
                for rad in range(radius+1):
                    pd_dict_one = {'raidus%d_%d'%(rad, idx): val for idx, val in enumerate(val_radius[idx])}
                    pd_dict.update(pd_dict_one)
                ret_db[key] = pd_dict
            else:
                ret_db[key] = vals

        pd_db = pd.DataFrame.from_dict(ret_db, orient='index')
        pd_db['input_smiles'] = pd_smiles
        return pd_db


def GetRepForSmiles(smiles, radius, model):
        
    sentence = GetFeatureSentences(smiles, radius=radius)
    
    # parse the sentence and get representation for each atoms
    vals = dict()
    
    for num, word in enumerate(sentence.split(' ')):
        
        if num%(radius+1)==0:
            word_list = []
            
        atom_idx = num // (radius+1)
        # Only words in the vocaburary can be persed
        if model.get_word_id(word) != -1:
            word_list.append(model[word])
        else:
            print('Found word that are not in the dictionary')
            return None
        vals[atom_idx] = word_list
    return vals

def GetStatsCatFiles(in_flist, out_file, remove_input):
    """
    Statistics (frequency count)
    """
    count_words = defaultdict(int)
    with open(out_file, 'w') as out:
        for in_f in in_flist:
            with open(in_f, 'r') as infp:
                for line in infp:
                    words = line.strip().split(' ')
                    for word in words:
                        count_words[word] += 1
                    out.write(line)

    # output statistics
    outf_stats = out_file.replace('.', '_stats.')
    pd_stats = pd.Series(count_words)
    pd_stats.sort_values(ascending=False, inplace=True)
    pd_stats.to_csv(outf_stats)

    if remove_input:
        for in_f in in_flist:
            os.remove(in_f)

def MakeInputWorker(folder_name, radius, smiles, njobs):
    # Input information for running the multithred
    set_smiles = np.array_split(smiles, njobs)
    return [[idx, folder_name, radius, smi] for idx, smi in enumerate(set_smiles)]

def GetFeatureSentences_worker(smiles_id):
    job_id = smiles_id[0]
    folder = smiles_id[1]
    radius = smiles_id[2]
    smiles = smiles_id[3]

    fp_hash = open('%s/hash_%d.txt'%(folder, job_id), 'w', buffering=1)

    for idx, smi in smiles.items():
        
        if Chem.MolFromSmiles(smi).GetNumAtoms() == 1:
            continue
        
        h_val = GetFeatureSentences(smi, radius)

        fp_hash.write('%s\n'%h_val)

    fp_hash.close()


def GetFeatureSentences(smiles, radius):
    
    mol = Chem.MolFromSmiles(smiles)

    # ECFP invariants for making a setntence
    feature_levels = CalcECFPFeatureLevels(mol, radius)
    # Making sentences (hash, invariants) for atoms on the path canonical smiles
    hash_sentence = ''
      
    for atom in mol.GetAtoms():
        
        for depth in range(radius+1):
            
            try:
                hash_sentence    += " {}".format(feature_levels[atom.GetIdx()][depth])
            except:
                continue


    # only start and end points
    hash_sentence = hash_sentence.strip()
    return hash_sentence
