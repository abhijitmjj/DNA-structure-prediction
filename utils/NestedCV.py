from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def gen_data(data: pd.DataFrame, RESAMPLING: bool=False):
    X, y = data.drop(labels="target", axis=1), data["target"]
    sss = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    #sss = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
    for train_index, val_index in sss.split(X, y):
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X.iloc[train_index,:], y.iloc[train_index])
        yield {"X_train": X_resampled if RESAMPLING else X.iloc[train_index,:],
               "y_train": y_resampled if RESAMPLING else y.iloc[train_index],
              "X_val": X.iloc[val_index,:], "y_val": y.iloc[val_index]}
        
def gen_data_for_tuningHP(data: dict, RESAMPLING: bool=False):
    
    X, y = data["X_train"], data["y_train"]
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    for train_index, val_index in sss.split(X, y):
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X.iloc[train_index,:], y.iloc[train_index])
        yield {"X_train": X_resampled if RESAMPLING else X.iloc[train_index,:],
               "y_train": y_resampled if RESAMPLING else y.iloc[train_index],
               "X_val": X.iloc[val_index,:], "y_val": y.iloc[val_index]}
    