<h1>Identification of metabolites in LC-MS untargeted datasets using machine learning</h1>

There are three folders: <strong>Data_Preprocessing</strong>, <strong>Data_Modelling</strong>, <strong>Data_Visualization</strong>.

* <strong>Data_Preprocessing</strong>: contains codes for getting molecular descriptors

* <strong>Data_Modelling</strong>: contains codes for building models

* <strong>Data_Visualization</strong>: contains codes for visulaizing results

<strong>Note</strong>: When using the codes in <strong>Data_Preprocessing</strong>, we need to install the rdkit environment. The specifc shell codes show as below:

```bash
conda create -c rdkit -n my-rdkit-env rdkit
```
```bash
conda activate my-rdkit-env
```
```bash
jupyter notebook
```
When using the codes in the other two folders, there is no need to implement in the rdkit environment.
