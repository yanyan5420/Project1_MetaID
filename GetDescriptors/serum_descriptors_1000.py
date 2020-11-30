#!/usr/bin/env python
# coding: utf-8

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np


def smiles_descriptors(data):
    '''
    This function is to get the descriptors of each SMILES, and get a new data frame
    
    Parameters
    ==========
    data: DataFrame
        a data frame containing metabolites and their corresponding SMILES
    
    Output
    ======
    data: DataFrame
        a updated data frame containing descriptors
    '''
    for desc, func in Descriptors.descList:
        data.loc[:,desc] = data.SMILES.apply(lambda x: func(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) is not None else 'nan')

    return data


data = pd.read_csv('serum_metabolites.csv', nrows=1000)
new_data = smiles_descriptors(data)
new_data.to_csv("serum_metabolite_descriptors_1000.csv")
