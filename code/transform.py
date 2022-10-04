from openeye.oechem import *
from openeye.oequacpac import *

def SmilesToOEGraphMol(smiles, strict=False):
    """
    wrapper function for making a new molecule from similes

    Add None if the molecule is not correct..
    """
    if smiles is None:
        return None

    mol = OEGraphMol()
    if strict:
        mol2 = OEGraphMol()
        if not OEParseSmiles(mol2, smiles, False, True):
            return None

    output = OESmilesToMol(mol, smiles)
    if not output:
        return None
    else:
        return mol

def DisconnectSalt(mol):
    """
    Following instructions on MOE's help webpage to create a similar
    dissociation between alkaline metals and organic atoms
    """
    metals = [3, 11, 19, 37, 55] #Alkaline: Li, Na, K, Rb, Cs
    organics = [6, 7, 8, 9, 15, 16, 17, 34, 35, 53] # C, N, O, F, P, S, Cl, Se, Br, I


    bondsToDel=[]
    for bond in mol.GetBonds():
        if bond.GetOrder()!= 1:
            continue
        a1 = bond.GetBgn()
        a2 = bond.GetEnd()

        if a1.GetFormalCharge()!= 0 or a2.GetFormalCharge()!=0:
                continue

        if a1.GetAtomicNum() in metals:
            if a2.GetAtomicNum() not in organics:
                continue
            if a1.GetHvyDegree()!=1:
                continue
            bondsToDel.append(bond)

        elif a2.GetAtomicNum() in metals:
            if a1.GetAtomicNum() not in organics:
                continue
            if a2.GetHvyDegree()!=1:
                continue
            bondsToDel.append(bond)

    for bond in bondsToDel:
        mol.DeleteBond(bond)

def RemoveIsotopeInformation(mol):
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)

def UndoProtonation(mol):
    """
    Undo the (de-)protonation of O and N performed by
    OEAssignImplicitHydrogens
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()==7 and atom.GetHvyValence()<4:
            atom.SetImplicitHCount(3-atom.GetHvyValence())
            atom.SetFormalCharge(0)

        if atom.GetAtomicNum()==16 and atom.GetHvyValence()==1:
            for nbor in atom.GetAtoms():
                    break
            if nbor.GetFormalCharge()<=0:
                atom.SetImplicitHCount(1)
                atom.SetFormalCharge(0)

        if atom.GetAtomicNum()==8 and atom.GetHvyValence()==1:
                for nbor in atom.GetAtoms():
                    break
                # nbor is the only neighbor of atom
                if nbor.GetAtomicNum()==16 and nbor.GetHvyValence() & 1:
                    bond=atom.GetBond(nbor)
                    db_flag=True
                    if nbor.GetHvyValence()==5 and not bond.IsAromatic():
                        bond.SetOrder(2)
                        atom.SetImplicitHCount(0)
                        atom.SetFormalCharge(0)
                        nbor.SetFormalCharge(0)
                else:
                    if nbor.GetFormalCharge()<=0:
                        atom.SetImplicitHCount(1)
                        atom.SetFormalCharge(0)
                        """ # only protonate [O-]X(=O)
                        for nnbor in nbor.GetAtoms():
                            if nnbor.GetAtomicNum()==8 and nbor.GetBond(nnbor).GetOrder()==2:
                                atom.SetImplicitHCount(1)
                                atom.SetFormalCharge(0)
                                break
                        """

def CountExplicitH(mol):
    ct = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()==1:
            ct +=1
    return ct

def GetNonChiralAromaticSmiles(mol):
    return OECreateSmiString(mol,OESMILESFlag_Canonical)

def Smiles2WahsedSmiles(smiles, return_nonstereo=False, ignore_warning=False):

    washed_mol, warning = WashOEGraphMol(SmilesToOEGraphMol(smiles), return_warning=True)
    if washed_mol is None:
        return None

    if warning and (not ignore_warning):
        print('Fail wash molecule: %s'%smiles)
        return None

    if return_nonstereo:
        return GetNonChiralAromaticSmiles(washed_mol)
    else:
        return OEMolToSmiles(washed_mol)


def WashOEGraphMol(org_mol, return_warning=False):
    """
    Wash molecules based on the Martin's script
    concept is to neutralize molecule as much as possible
    Note! even though warning messages appear, input molecule must be output
    If curation fail. They output "None" instead of original one. (changed 28. 07. 2021)
    input:
    ------
    inplace: default False

    """

    mol =  OEGraphMol(org_mol)

    # DisconnectSalt(mol)
    OEDeleteEverythingExceptTheFirstLargestComponent(mol) # weired name
    RemoveIsotopeInformation(mol)
    OESuppressHydrogens(mol)

    warning = False
    if CountExplicitH(mol):
        warning = True
        Warning("EXPLICIT HYDROGENS FOUND IN", OECreateIsoSmiString(mol),"AFTER SUPPRESS")

    UndoProtonation(mol)
    OEAssignFormalCharges(mol)

    if CountExplicitH(mol):
        warning = True
        Warning("EXPLICIT HYDROGENS FOUND IN", OECreateIsoSmiString(mol),"AFTER FC")

    OEMDLPerceiveParity(mol)
    OEMDLPerceiveBondStereo(mol)
    OEAssignAromaticFlags(mol,OEAroModel_OpenEye)

    if CountExplicitH(mol):
        warning = True
        Warning("EXPLICIT HYDROGENS FOUND IN", OECreateIsoSmiString(mol),"AFTER ARO")

    OEAssignHybridization(mol)

    if CountExplicitH(mol):
        warning = True
        Warning("EXPLICIT HYDROGENS FOUND IN", OECreateIsoSmiString(mol),"AFTER HYBRY")

    is_valid = True
    # check the valency rules...
    if OECount(mol, OENotAtom(OEIsValidMDLAtomValence())) > 0:
        Warning("Atoms with invalid valency exist.")
        is_valid = False

    # added on 28.07.2021
    if not is_valid:
        mol = None

    if not return_warning:
        return mol
    else:
        return mol, warning
