import os
import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import KFold


problem_title = 'Prediction of airbnb price'
_target_column_name = 'price' 
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow

class Airbnb(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'regressor', 'train.csv']):
        super(Airbnb, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names

workflow =  Airbnb()

# define the score (specific score for the FAN problem)
class Airbnb_error(BaseScoreType):

    def __init__(self, name='airbnb error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        loss = np.mean(np.abs((y_true - y_pred) / (y_true))) * 100
        
        return loss

score_types = [
    Airbnb_error(name='airbnb error'),
]

def get_cv(X, y):
    cv = KFold(n_splits=8, random_state=42)
    return cv.split(X,y)

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)