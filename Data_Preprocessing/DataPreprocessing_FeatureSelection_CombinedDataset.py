#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


# In[ ]:


def smiles_descriptors(data):
    '''
    This function is to get the descriptors of each SMILES, and get a new data frame
    =========
    Parameter:
    data: DataFrame
        a data frame containing metabolites and their corresponding SMILES
    =========
    
    Output: a updated data frame containing descriptors
    '''
    
    for desc, func in Descriptors.descList:
        data.loc[:,desc] = data.SMILES.apply(lambda x: func(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) is not None else 'nan')

    return data


# In[ ]:


def filter_std(data, threshold):
    '''
    This function is to filter descriptors by standard deviations
    =========
    Parameter:
    
    data: DataFrame
        a data frame containing metabolites and their corresponding descriptors
    
    threshold: float
    =========
    
    Output: a filtered data frame containing descriptors
    '''
    
    drop_columns = []
    
    for col in data.columns[6:]:
        
        if data[col].std() <= threshold:
            drop_columns.append(col)
    
    data.drop(drop_columns, axis=1, inplace=True)
    
    return data


# In[ ]:


def get_removed_correlation(df, feature_list, feature, threshold, remove_list):
    '''
    This function is to filter descriptors by Pearson correlation
    =========
    Parameter:
    
    df: DataFrame
        a data frame containing metabolites and their corresponding descriptors
    
    feature_list: List
        the list of features that are filtered by std
    
    feature: String
        a molecular descriptor
    
    threshold: float
    
    remove_list: List
        the list of descriptors that need to be deleted
    =========
    
    Output: a filtered data frame containing descriptors
    '''
    
    if len(feature_list) == 0:
        return remove_list

    for f in df[feature][df[feature]>=threshold].index:
        if f != feature and f not in remove_list:
            remove_list.append(f)

    feature_list.remove(feature)

    df.drop(index=feature, inplace=True)
    if len(df) != 0:
        feature = df[feature][df[feature]==min(df[feature])].index[0]

    return get_removed_correlation(df, feature_list, feature, threshold, remove_list)


# In[ ]:


def get_training_features(input_filename, std_threshold, r_threshold, output_filename):
    '''
    This function is to filter descriptors for training dataset
    =========
    Parameter:
    
    input_filename: String
        input file name
    
    std_threshold: float
        the threshold of filtering by std
    
    r_threshold: float
        the threshold of filtering by Pearson correlation
    
    output_filename: String
        output file name
    =========
    '''
    
    data = pd.read_csv(input_filename)
    
    # get descriptors based on SMILES
    data = smiles_descriptors(data)
    
    # filter descriptors with std
    data = filter_std(data, std_threshold)
    print(data.shape)
    
    # filter descriptors with r2
    r2_df = data.iloc[:,6:].corr()**2
    feature_list = list(r2_df.columns)
    start_feature = 'MolLogP'
    remove_list = []
    remove_list = get_removed_correlation(r2_df, feature_list, start_feature, r_threshold, remove_list)
    
    data.drop(remove_list, axis=1, inplace=True)
    data.sort_values(['rtmed'], inplace=True)
    data.reset_index(inplace=True, drop=True)
    
    data.to_csv(output_filename, index=False)
    
    return


# In[ ]:


# input_filename = "combined_training_data.csv"
# output_filename = "combined_train_with_features.csv"
# std_threshold = 0.01
# r_threshold = 0.96
# get_training_features(input_filename, std_threshold, r_threshold, output_filename)


# In[ ]:


def get_validation_features(input_filename, output_filename, train_filename):
    '''
    This function is to get descriptors for validation dataset
    =========
    Parameter:
    
    input_filename: String
        input file name
    
    output_filename: String
        output file name
        
    train_filename: String
        training data file name
    =========
    '''
    
    train_data = pd.read_csv(train_filename)
    descriptors = list(train_data.columns[6:])
    
    data = pd.read_csv(input_filename)
    
    for desc, func in Descriptors.descList:
        if desc in descriptors:
            data.loc[:,desc] = data.SMILES.apply(lambda x: func(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) is not None else 'nan')
    
    data.sort_values(['rtmed'], inplace=True)
    data.reset_index(inplace=True, drop=True)
    
    data.to_csv(output_filename, index=False)
    
    return


# In[ ]:


# input_filename = "combined_validation_data.csv"
# output_filename = "combined_valid_with_features.csv"
# train_filename = "combined_train_with_features.csv"
# get_validation_features(input_filename, output_filename, train_filename)


# In[ ]:


def get_lipids_features(input_filename, output_filename, train_filename):
    '''
    This function is to get descriptors for NIST Lipids dataset
    =========
    Parameter:
    
    input_filename: String
        input file name
    
    output_filename: String
        output file name
        
    train_filename: String
        training data file name
    =========
    '''
    
    train_data = pd.read_csv(train_filename)
    descriptors = list(train_data.columns[6:])
    
    data = pd.read_csv(input_filename)
    
    for desc, func in Descriptors.descList:
        if desc in descriptors:
            data.loc[:,desc] = data.SMILES.apply(lambda x: func(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) is not None else 'nan')
    
#     data.sort_values(['rtmed'], inplace=True)
#     data.reset_index(inplace=True, drop=True)
    
    data.to_csv(output_filename, index=False)
    
    return


# In[ ]:


# input_filename = "Lipids_NIST_Data.csv"
# output_filename = "Lipids_NIST_Data_with_features.csv"
# train_filename = "combined_train_with_features.csv"
# get_lipids_features(input_filename, output_filename, train_filename)


# In[ ]:


# lipid_df = pd.read_csv("Lipids_NIST_Data_with_features.csv")


# In[ ]:


# lipid_df


# In[ ]:




