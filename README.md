# DNA-structure-prediction
Accurate Prediction of DNA structure from its sequence using Machine Learning
Gupta, Abhijit; Kulkarni, Mandar; Mukherjee, Arnab (2020): Accurate Prediction of B-form/A-form DNA Conformation Propensity from Primary Sequence: A Machine Learning and Free energy Handshake. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.12599633.v3   
[![DOI](https://zenodo.org/badge/301697564.svg)](https://zenodo.org/badge/latestdoi/301697564)
# Workflow diagram
The **src** directory contains the different machine learning algorithms that we tested using 5-fold nested cross validation strategy
<p align="left">
  <img src="workflow_diagram.png" width="500" title="hover text">
</p>


# Data 

The **data** directory contains our curated dataset - `curated_data.csv`. We have also provided the dataset in pickle format inside the `./data/pkl` directory. Inside the **data** directory, we have provided "Experimental conditions.xls". It contains the different experimental conditions under which our curated set of sequences where obtained. 

# Utilities 
The **utils** directory has two modules "NestedCV" and "Evaluator". "NestedCV" contains two generator functions, which provide implementation of nested cross-validation. The gen_data implements the "outer loop" by doing k-fold Stratified cross-validation. It also provides the option for performing SMOTE+Tomek on the training data, which can be turned on by setting the parameter `RESAMPLING=True`. At each iteration it yields a dictionary of a pair of "train" and "test" samples.
The "inner loop" is implemented by another generator function named "gen_data_for_tuningHP". It takes the "train" data from an iteration of the outer loop and splits it into "inner train data" and "validation" data. It then performs the inner stratified k-fold cross-validation for tuning hyperparameters of an algorithm. 

The next module, **Evaluator** is used for training and evaluation. It generates the "ROC-AUC" and "Precision-recall" graphs and outputs a dataframe that contains the result of performing nested cross validation. It also performs the classification by choosing the optimal threshold and then converting the class probabilities into the class labels.

We have provided the Notebook files for each model inside the **src** directory. To ensure reproducability, we have provided the tuned hyperparameters and the data splits that we used. To reproduce our results, you can either use the seeds we used in the notebook or use the splits of data given in **/data/train_test_folds**. By default, we have set the seed to **42**, which gives the same split of data that is provided in "train_test_folds" directory.

# Dependencies
We have listed below the key dependencies required for running different ML algorithms.

dependencies:
You can install them by running `conda install -c conda-forge <name-of-the-package>
  - conda
  - python=3
  - conda-build
  - lightgbm
  - jupyter
  - nodejs
  - imbalanced-learn
  - more-itertools
  - toolz
  - matplotlib
  - seaborn
  - dask
  - flask
  - optuna
  - regex
  - dill
  - shap
  - scikit-learn
  - numpy
  - xlrd  

We have also provided the `environment.yml` file for replicating our conda environment. If you do not intend to install all packages listed below separately, then you can create the environment from the environment.yml file:
`conda env create -f environment.yml`

# How do I run/test different models? 

For the ease of use, we have provided each model separately. Simply go to the **src** directory and run the Jupyter notebooks to try out different approaches. 
