#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


def get_meta_pos_neg(lcms_data_file, annotate_data_file, sheet):
    '''
    This functio is to get the metabolite data with one feature in positive mode or negative mode
    
    Parameters
    ==========
    lcms_data_file: xlsx
        an Excel file containing all the sample, feature and intensity information in positive or negative mode
    annotate_date_file: xlsx
        an Excel file containing the annotated metabolites in positive and negative modes
    sheet: str
        the sheet name for positive mode data or negative mode data
    
    Output
    ======
    DataFrame
        a data frame that removing features with n.d. and keeping the features with the highest intensity
    '''
    
    lcms_data = pd.read_excel(lcms_data_file, index_col=0)
    annotate_data = pd.read_excel(annotate_data_file, sheet_name=sheet)
    
    # remove the rows whose AIRWAVE1_MZRT_str value is n.d.
    filter_annotate = annotate_data[annotate_data['AIRWAVE1_MZRT_str']!='n.d.']
    filter_annotate = filter_annotate.sort_values(by=['Metabolite'])
    filter_annotate = filter_annotate.reset_index(drop=True)
    
    # get the intensity for each row
    fimed_dict = dict(lcms_data.loc['fimed',:])
    filter_annotate.loc[:,'fimed'] = filter_annotate['AIRWAVE1_MZRT_str']                                    .apply(lambda x: fimed_dict[x] if x in fimed_dict else -1)
    
    # use the feature with the highest intensity to represent each metabolite
    idx = filter_annotate.groupby(["Metabolite"])["fimed"].transform(max)==filter_annotate["fimed"]
    meta_data = filter_annotate[idx].reset_index(drop=True)
    
    return meta_data



def find_pos_neg_overlap(meta_pos, meta_neg):
    '''
    This function is to find the overlap data between positive and negative modes and save them into Excel files
    
    Parameters
    ==========
    meta_pos: DataFrame
        a data frame containing metabolites information in positive mode
    meta_neg: DataFrame
        a data frame containing metabolites information in negative mode
    '''
    
    pos_neg_overlap = pd.merge(meta_pos, meta_neg, how='inner', on='Metabolite')                    .loc[:,['Metabolite','IonType_x','AIRWAVE1_MZRT_str_x','RefMet_Standardized_name _x','fimed_x',                            'IonType_y','AIRWAVE1_MZRT_str_y','RefMet_Standardized_name _y','fimed_y']]
    
    pos_only = meta_pos.merge(meta_neg, on='Metabolite', how = 'outer', indicator=True).loc[lambda x : x['_merge']=='left_only']
    pos_only = pos_only.loc[:,['Metabolite', 'LOA_x', 'Theoretical_mz_x', 'IonType_x', 'AIRWAVE1_MZRT_str_x', 'AIRWAVE1_MZmed_x',                               'AIRWAVE1_RTmed_x','RefMet_Standardized_name _x', 'RefMet_Formula_x', ' RefMet_Exact_mass_x',                               'RefMet_Super_class_x', 'RefMet_Main_class_x', 'RefMet_Sub_class_x','Database_CHEBI_ID_x', 'fimed_x', '_merge']]
    neg_only = meta_pos.merge(meta_neg, on='Metabolite', how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
    neg_only = neg_only.loc[:,['Metabolite', 'LOA_y', 'Theoretical_mz_y', 'IonType_y','AIRWAVE1_MZRT_str_y', 'AIRWAVE1_MZmed_y',                               'AIRWAVE1_RTmed_y','RefMet_Standardized_name _y', 'RefMet_Formula_y', ' RefMet_Exact_mass_y',                               'RefMet_Super_class_y', 'RefMet_Main_class_y', 'RefMet_Sub_class_y','Database_CHEBI_ID_y', 'fimed_y', '_merge']]
    
    pos_neg_overlap.to_excel("pos_neg_overlap.xlsx", index=False)
    pos_only.to_excel("pos_metabolites.xlsx", index=False)
    neg_only.to_excel("neg_metabolites.xlsx", index=False)
    
    return


# positive mode data
lcms_pos_file = "LC-MS_data/POS_Data.xlsx"
annotate_data_file = "LC-MS_annotations_AWI.xlsx"
sheet1 = "SLPOS"
meta_pos = get_meta_pos_neg(lcms_pos_file, annotate_data_file, sheet1)

# negative mode data
lcms_neg_file = "LC-MS_data/NEG_Data.xlsx"
annotate_data_file = "LC-MS_annotations_AWI.xlsx"
sheet2 = "SLNEG"
meta_neg = get_meta_pos_neg(lcms_neg_file, annotate_data_file, sheet2)

# get the overlap between positive and negative modes
find_pos_neg_overlap(meta_pos, meta_neg)
