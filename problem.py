import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit


problem_title = 'AirBnB price prediction'
_target_column_name = 'price' 
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow

workflow = FeatureExtractorRegressor()

score_types = [
    rw.score_types.RMSE(name='rmse', precision=3),
]

def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=7)
    return cv.split(X,y)

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False,
                       compression='zip')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'train.csv.zip'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'test.csv.zip'
    return _read_data(path, f_name)