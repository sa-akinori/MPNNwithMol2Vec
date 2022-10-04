'''
Created on Feb 27, 2014
@author: vogtm
Closely follows Rogers D, Hahn M. Extended-Connectivity Fingerprints. JChemInfModel 2010, 50, 742-754.
'''

from openeye.oechem import *
from transform import SmilesToOEGraphMol
from collections import namedtuple
from openeye.oechem import *
from numpy import int64,int32,integer

class Fingerprint(object):
    def __init__(self, name, version):
        self.name_ = name
        self.version_ = version

    def name(self):
        return self.name_

    def fullname(self):
        if self.version_:
            return self.name_ + "_" + self.version_
        else:
            return self.name_

    def featureList(self, mol):
        raise NotImplementedError

    def bitString(self, mol):
        raise NotImplementedError

    def hasBitString(self):
        return False


def myHashLong(obj):
    Int=int64
    if not obj:
        return 0
    if isinstance(obj,str):
        mult = Int(1000003)
        value = Int(ord(obj[0]) << 7)
        for char in obj:
            value = (mult*value) ^ Int(ord(char))
        value = value ^len(obj)
    elif isinstance(obj,int) or isinstance(obj,integer) or isinstance(obj,int):
        value = Int(obj&0xffffffff)
        obj = int(obj) >> 32
        while obj!=0 and obj !=-1:
            value = value ^ (obj&0xffffffff)
            obj = obj >> 32
    elif isinstance(obj,tuple):
        mult = Int(1000003)
        n = Int(len(obj))
        value = Int(0x345678)
        for item in obj:
            value = mult*(value ^ myHashLong(item))
            mult += Int(82520)+n+n
        value += Int(97531)
    else:
        #print obj
        #print type(obj)
        raise RuntimeError("Hash not implemented for %s"%type(obj))
    return value

def myHash(obj):
    return int32(myHashLong(obj) & 0xffffffff)
    """
    if value & 0x80000000:
        return -int32(-value)
    else:
        return int32(value)
    """

emptyset = frozenset()
Invariant = namedtuple('Invariant',['lvl','bondSet','pattern','hash'])

def ecfpInvariant(atom):
    return Invariant(lvl = 0,
                    bondSet=emptyset,
                    pattern="[#%dD%dv%d%+dH%d%s]" % (atom.GetAtomicNum(),
                                                    atom.GetHvyDegree(),
                                                    atom.GetValence(),
                                                    atom.GetFormalCharge(),
                                                    atom.GetExplicitHCount() + atom.GetImplicitHCount(),
                                                    "R" if atom.IsInRing() else "R0",
                                                    # 1 if atom.IsAromatic() else 0, # not in original publication
                                                    ),
                    hash=myHash("%d%02d%02d%+d%d%d"%(atom.GetHvyDegree(),
                            atom.GetHvyValence(),
                            atom.GetAtomicNum(),
                            atom.GetFormalCharge(),
                            atom.GetExplicitHCount()+atom.GetImplicitHCount(),
                            1 if atom.IsInRing() else 0,
                            # 1 if atom.IsAromatic() else 0, # not in original publication
                            )))

def getBranches(patt):
    branches = []
    bgn = True
    for idx in range(len(patt)):
        if bgn:
            if not patt[idx] == '(':
                branches.append(patt[idx:])
                break
            else:
                start = idx + 1
                lvl = 1
                bgn = False
        else:
            if patt[idx] == ')':
                lvl -= 1
            elif patt[idx] == '(':
                lvl += 1
            if lvl == 0:
                branches.append(patt[start:idx])
                bgn = True
    return branches

def asPrunedPattern(patt, remove, order):
    bond = '.-=#:'[order]
    idx = patt.index(']')
    branches = getBranches(patt[idx + 1:])
    newBranches = []
    found = False
    for branch in branches:
        if not found and branch[0] == bond and startingAtom(branch[1:]) == remove:
            found = True
            continue
        newBranches.append(branch)
    """
    if not found and len(branches)>1:
        print "ERROR"
        print patt,remove,order
        print bond
        print branches
        for branch in branches:
            print branch[0],bond
            print startingAtom(branch[1:]),remove
            #if not found and branch[0]==bond and startingAtom(branch[1:])==remove:
        assert(False)
    """
    return patt[:idx + 1] + "".join("(%s)" % x for x in newBranches[:-1]) + "".join(newBranches[-1:])


def asRecursivePattern(patt):
    idx = patt.index(']')
    return patt[:idx] + ";$(*" + patt[idx + 1:] + ")]"

def startingAtom(patt):
    idx=patt.index(']')
    return patt[:idx+1]


lastPatternDict = {}
def ecfcFingerprint4Mol2Vec(mol, radius, invariant=ecfpInvariant, patterns=lastPatternDict):
    """
    Add unsorted option to get atom order inavriants
    """
    graph = dict()
    # key: atom index
    # value: list of neighbor tuples (atomIndex,bondIndex,bondOrder)
    radius = int(radius)
    for atom in mol.GetAtoms():
        # no need to calculate R group at level 0
        nbors = []
        for bond in atom.GetBonds():
            nbr = bond.GetNbr(atom)

            if nbr.GetAtomicNum() != 1:
                order = bond.GetOrder() if not bond.IsAromatic() else 4
                nbors.append((nbr.GetIdx(), bond.GetIdx(), order))
        graph[atom.GetIdx()] = nbors

    level0 = dict(
        (atom.GetIdx(), invariant(atom)) for atom in mol.GetAtoms())

    # featInfo(level0)
    # Invariant dictionary
    # key: atom index
    # value: invariant, bond subset covered
    featureLevels = [level0]  # growing array of feature sets: 1 per iteration
    bondSetFeatures = dict()  # keeps track of invariants for each bond set

    for lvl in range(1,radius+1):
        old = featureLevels[-1]
        new = dict()
        for a, prev in list(old.items()):
            nb = []
            # list of neighbor invariants: (bond order,nbor invariant)
            bondSet = set(prev.bondSet)
            for nbor, bond, order in graph[a]:
                bondSet.update(old[nbor].bondSet)
                bondSet.add(bond)
                nb.append((order, old[nbor].hash,old[nbor].pattern))
            nb.sort() # this is imporltant to assure the correct order
            bondSet = frozenset(bondSet)
            newInv = myHash((lvl, prev.hash, tuple((x,y) for x,y,_ in nb)))
            if lvl==1:
                nborPatterns = [".-=#:"[order]+patt for order,_,patt in nb]
                newPatt = prev.pattern+"".join("(%s)"%x for x in nborPatterns[:-1])+"".join(nborPatterns[-1:])
            elif lvl==2:
                # Using non-recursive patterns for radius 2 will fail for rings of size 4 or less because
                # some atoms appears more than once in the pattern (as neighbors of different atoms)
                center = startingAtom(prev.pattern)
                nborPatterns = [".-=#:"[order]+asPrunedPattern(patt,center,order) for order,_,patt in nb]
                newPatt = center+"".join("(%s)"%x for x in nborPatterns[:-1])+"".join(nborPatterns[-1:])
            else:
                center = startingAtom(prev.pattern)
                nborPatterns = [".-=#:"[order]+asRecursivePattern(patt) for order,_,patt in nb]
                newPatt = center+"".join("(%s)"%x for x in nborPatterns[:-1])+"".join(nborPatterns[-1:])

            new[a] = Invariant(lvl=lvl,bondSet=bondSet,hash=newInv,pattern=newPatt)

        featureLevels.append(new)

    # level information is also stored
    features, patterns = list(), list()
    for featureLevel in featureLevels:
        features.extend([x.hash for x in featureLevel.values()])
        patterns.extend([((idx, x.lvl),(x.hash, x.pattern)) for idx, x in featureLevel.items()])

    features = sorted(features)
    patterns = dict(patterns)

    return features, patterns

class Ecfp4Mol2Vec(Fingerprint):
    def __init__(self, diameter, invariant, name, version):
        super(Ecfp4Mol2Vec, self).__init__(name, version)
        self.radius = diameter/2
        self.invariant = invariant

    def featureLevels(self,mol):
        hash, patterns =  ecfcFingerprint4Mol2Vec(mol, self.radius, self.invariant)
        return hash, patterns

def removeIsotopeInformation(mol):
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)

def PrepareMol(mol_in):
    """
    Eliminate hydrogen explicit information
    """
    mol = OEGraphMol(mol_in)
    OESuppressHydrogens(mol)
    removeIsotopeInformation(mol)
    OEAssignHybridization(mol)
    OEAssignAromaticFlags(mol, OEAroModel_OpenEye)

    return mol

def CalcECFPFeatureLevels4Mol2Vec(oemol, diameter=4, prepare_mol=True):
    """
    atom index is the key of the feature level
    """
    ecfp = Ecfp4Mol2Vec(diameter, ecfpInvariant, 'ECFP%d'%diameter, 'v3')
    if prepare_mol:
        mol = PrepareMol(oemol)
    else:
        mol = oemol

    return ecfp.featureLevels(mol) # feature level list is return.
