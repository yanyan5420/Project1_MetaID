#!/usr/bin/env python
# coding: utf-8

# import needed libraries
import xml.etree.ElementTree as ET
import csv


"""
This function is to parse xml file and select needed data as a dictionary.
Input: XML file name
Output: a dictionary containing needed data
"""
def xml_to_dict(filename):
    
    # use element tree to parse the XML file
    tree = ET.parse(filename)
    root = tree.getroot()
    # since this xml file contains namespaces, for convinience, make a dictionary to replace the namespace
    namespace = {'metabolite': 'http://www.hmdb.ca'}
    
    # initialize a empty dictionary to place the data
    meta_dict = {}
    N = 0
    
    # for each child in the root, run the following loop
    for meta in root.findall('metabolite:metabolite', namespace):
        
        # get the name, hmdb_id, smiles of the metabolite
        NAME = meta.find('metabolite:name', namespace)
        HMDB_ID = meta.find('metabolite:accession', namespace)
        SMILES = meta.find('metabolite:smiles', namespace)
        
        # save these data into the dictionary
        info_dict = {}
        info_dict['NAME'] = NAME.text
        info_dict['HMDB_ID'] = HMDB_ID.text
        info_dict['SMILES'] = SMILES.text
        meta_dict[N] = info_dict

        N += 1
    
    return meta_dict



"""
This function is to save the dictionary into a CSV file
Input: the dictionary containing metabolite information, the CSV filename
"""
def dict_to_csv(meta_dict, csv_file):
    
    # turn the metabolite dictionary into a list of dictionary
    meta_list = list(meta_dict.values())
    # get the column names
    csv_columns = list(meta_list[0].keys())
    
    # write the list of dictionary into a CSV file
    with open(csv_file, 'w') as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames = csv_columns)
        writer.writeheader()
        
        for data in meta_list:
            writer.writerow(data)


"""
This function is to parse the XML file into a CSV file
Input: XML filename, CSV filename 
"""
def main(filename, csv_file):
    
    meta_dict = xml_to_dict(filename)
    dict_to_csv(meta_dict, csv_file)



filename = 'serum_metabolites.xml'
csv_file = 'serum_metabolites.csv'
main(filename, csv_file)



#import pandas as pd
#data = pd.read_csv(csv_file)
#data.head()






