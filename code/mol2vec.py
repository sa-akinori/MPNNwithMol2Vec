import pandas as pd
import numpy as np
import multiprocessing
import os
import fasttext
from transform import Smiles2WahsedSmiles, SmilesToOEGraphMol
from fingerprints import CalcECFPFeatureLevels4Mol2Vec
from utility import MakeFolder, MakeLogFP, WriteMsgLogStdout, AssertTerminate
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
    def MakeFastTextInputFile(pd_smiles, folder_name='mol2vec', radius=1, use_hash=True, njobs=10, debug=False, use_curated_smiles=False):
        """
        input_smiles: Skipt the curation part for saving the time (or reuse already curated smiles file)
        """

        folder = MakeFolder(folder_name)
        logFP = MakeLogFP('%s/mol2vec.log'%folder, add_timestamp=True)

        WriteMsgLogStdout(logFP, 'The number of input SMILES:\t%d'%len(pd_smiles))

        # Wash input molecules
        if not use_curated_smiles:
            WriteMsgLogStdout(logFP, 'SMILES curation and NonstereoAromacitSmiles conversion starts')


            split_smiles  = np.array_split(pd_smiles, njobs)
            if debug:
                washed_smiles = pd_smiles
            else:
                pool = multiprocessing.Pool(njobs)
                washed_smiles = pd.concat(list(pool.map(lSmiles2WashedSmiles, split_smiles)))
                pool.close()
                pool.join()

            WriteMsgLogStdout(logFP, 'Done Wash')
            washed_smiles.dropna(inplace=True)
            WriteMsgLogStdout(logFP, 'The number of passed Washing:\t%d'%len(washed_smiles))
            # Drop duplicates
            washed_smiles.drop_duplicates(inplace=True)
            WriteMsgLogStdout(logFP, 'The number of unique SMILES:\t%d'%len(washed_smiles))

            # Output clean smiles
            washed_smiles.to_csv('%s/input_smiles.txt'%folder, sep='\t') # series

        else:
            WriteMsgLogStdout(logFP, 'Skip the SMILES curation.')
            washed_smiles = pd_smiles.drop_duplicates()

        # Make the Sentence for them
        WriteMsgLogStdout(logFP, 'Making the FastText Input Files')
        WriteMsgLogStdout(logFP, 'Radius:\t%d'%radius)
        input_worker = MakeInputWorker(folder, radius, washed_smiles, njobs)

        if debug:
            a, b= GetFeatureSentences(washed_smiles.iloc[0], radius)
            print(100)
        else:
            pool = multiprocessing.Pool(njobs)
            pool.map(GetFeatureSentences_worker, input_worker)
            pool.close()
            pool.join()

            # cat the hash files (remove the current files later) / append statistics
            flist_h = ['%s/hash_%d.txt'%(folder, id) for id in range(njobs)]
            GetStatsCatFiles(flist_h, '%s/hash_sentence.txt'%folder, remove_input=True)

            flist_p = ['%s/patt_%d.txt'%(folder, id) for id in range(njobs)]
            GetStatsCatFiles(flist_p, '%s/pattern_sentence.txt'%folder, remove_input=True)

    @staticmethod
    def RunFastText(infile, outfile, ws=10, dim=100, minn=0, maxn=0, neg=5, epoch=20, minCount=1):
        model = fasttext.train_unsupervised(infile, model='skipgram', ws=ws, dim=dim, minn=minn, maxn=maxn,
                                            neg=neg, epoch=epoch, minCount=minCount) # off n-grams

        model.save_model(outfile)

    @staticmethod
    def GetRepresentation(pd_smiles, model_file, use_hash=True, radius=1, return_atomwise=False):
        """
        IF return_atomwise is false, all the vectors are summed to generated fixed-size matrix
        Otherwise the return contains the atom-wise information of mol2vec
        """

        AssertTerminate(isinstance(pd_smiles, pd.Series))
        model = fasttext.load_model(model_file)

        # wash the molecules
        washed_smiles = lSmiles2WashedSmiles(pd_smiles)

        # get the setntence
        ret_db = OrderedDict()

        for key, smiles in washed_smiles.items():
            vals = GetRepForSmiles(smiles, use_hash=use_hash, radius=radius, model=model)
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
        pd_db['input_washed_smiles'] = washed_smiles
        return pd_db


def GetRepForSmiles(smiles, radius, use_hash, model):
    h_val, p_val = GetFeatureSentences(smiles, radius=radius, return_atomidx=True)

    # which type is used for getting representations
    sentence = h_val if use_hash else p_val
    # parse the sentence and get representation for each atoms
    vals = dict()
    for atom_idx in sentence:
        word_list = []
        for word in sentence[atom_idx].split(' '):
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
    fp_patt = open('%s/patt_%d.txt'%(folder, job_id), 'w', buffering=1)

    for idx, smi in smiles.items():
        h_val, p_val = GetFeatureSentences(smi, radius=radius)
        fp_hash.write('%s\n'%h_val)
        fp_patt.write('%s\n'%p_val)

    fp_hash.close()
    fp_patt.close()

def lSmiles2WashedSmiles(pd_smiles):
    # nonstereo aromatic smiles (for the current ECFP implementation)
    return pd_smiles.apply(lambda x: Smiles2WahsedSmiles(x, return_nonstereo=True, ignore_warning=True))


def GetFeatureSentences(smiles, radius, return_atomidx=False):
    washed_mol = SmilesToOEGraphMol(smiles)
    diameter = 2*radius

    # ECFP invariants for making a setntence
    feature_levels = CalcECFPFeatureLevels4Mol2Vec(washed_mol, diameter=diameter)
    # Making sentences (hash, invariants) for atoms on the path canonical smiles
    hash_sentence = ''
    pattern_sentence = ''
    hash_dict    = defaultdict(str)
    pattern_dict = defaultdict(str)

    for atom in washed_mol.GetAtoms():
        # if atom.GetAtomicNum() == 1:
        #     continue
        if return_atomidx:
            for depth in range(radius+1):
                hash_dict[atom.GetIdx()]    +=" {}".format(feature_levels[1][(atom.GetIdx(), depth)][0])
                pattern_dict[atom.GetIdx()] +=" {}".format(feature_levels[1][(atom.GetIdx(), depth)][1])

            hash_dict[atom.GetIdx()]    = hash_dict[atom.GetIdx()].strip()
            pattern_dict[atom.GetIdx()] = pattern_dict[atom.GetIdx()].strip()

        else:
            for depth in range(radius+1):
                hash_sentence    += " {}".format(feature_levels[1][(atom.GetIdx(), depth)][0])
                pattern_sentence += " {}".format(feature_levels[1][(atom.GetIdx(), depth)][1]) # space separation.... lets see


    # only start and end points
    hash_sentence = hash_sentence.strip()
    pattern_sentence = pattern_sentence.strip()

    if return_atomidx:
        return hash_dict, pattern_dict
    else:
        return hash_sentence, pattern_sentence
