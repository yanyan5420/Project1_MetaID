#!/usr/bin/env python
# coding: utf-8


# import library to read Excel files
import pandas as pd
import functools, time


# def metric(func):
    
#     @functools.wraps(func)
#     def wrapper(*args, **kw):
        
#         start = time.time()
#         result = func(*args, **kw)
#         end = time.time()
#         print("function %s executed in %f s" % (func.__name__, end-start))
        
#         return result
    
#     return wrapper


# @metric
def merge_data(intensity_file, sample_file, feature_file, output_file, 
               filt_feature_columns, feature_name_column, sample_ID_column):
    """
    This function is to merge the intensity, sample, and feature files into one single file.

    Parameters 
    ==========
    intensity_file: str
        intensity file name.

    sample_file: str
        sample file name.

    feature_file: str
        feature file name.

    output_file: str
        the desired output file name.

    filt_feature_columns: list of str
        the needed feature column names

    feature_name_column: str
        the name of column containing feature names

    sample_ID_column: str
        the name of column containing sample ID
    ==========
    """
    
    # read input files
    intensity = pd.read_excel(intensity_file, header=None)
    sample = pd.read_excel(sample_file)
    feature = pd.read_excel(feature_file)
    
    # filter feature columns
    filt_feature = feature.loc[:,filt_feature_columns]
    filt_feature = filt_feature.set_index(feature_name_column).T
    
    # combine intensity and sample ID
    intensity.set_index(sample[sample_ID_column], inplace=True)
    intensity.columns = filt_feature.columns
    
    # merge intensity with sample ID and filtered features
    df = filt_feature.append(intensity)
    df = df.rename_axis(None, axis = 1)
    
    # write to a Excel file
    df.to_excel(output_file)


intensity_file = "Airwave1xcms_SLPOS_scaled_Data.xlsx"
sample_file = "Airwave1xcms_SLPOS_scaled_SampleInfo.xlsx"
feature_file = "SLPOS_MZRTstring_Airwave1_allBatches.xlsx"
output_file = "LCMS_data.xlsx"
filt_feature_columns = ["MZRT_str", "InitialVarNr", "rtmed", "mzmed", "fimed"]
feature_name_column = "MZRT_str"
sample_ID_column = "ICID"

merge_data(intensity_file, sample_file, feature_file, output_file, 
           filt_feature_columns, feature_name_column, sample_ID_column)
