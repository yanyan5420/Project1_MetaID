'''
Run descriptors with MPI

First, install mpi4py in the rdkit environment, and run the following command:
	
	conda install -c anaconda mpi4py

Then, run this .py file using following command:

	mpiexec -n numprocs python3 pyfile [arg]

For example:

	mpiexec -n 8 python3 serum_descriptors_mpi.py

'''

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


data = pd.read_csv('serum_metabolites.csv')

s = time.time()

# allocate data size for each rank
g = int(len(data) / size)
start_index = rank*g
if rank == size-1:
	end_index = len(data)
else:
	end_index = (rank+1)*g
local_data = data.iloc[start_index:end_index,:]


for desc, func in Descriptors.descList:
    local_data[desc] = local_data.SMILES.apply(lambda x: func(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) is not None else 'nan')

local_data.to_csv("descriptors_row_rank"+str(rank)+'.csv')

comm.Barrier()
e = time.time()
if rank==0:
	print("TIME is ", e-s, "!!!!!!!!!!!!!!!!!")
