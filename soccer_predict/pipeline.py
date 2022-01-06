import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score#, GridSearchCV
#from sklearn.metrics import make_scorer

def data_split(data, list_y, list_X, classification):
    X = data.drop(columns=list_X)
    y = data[list_y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if classification:
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        
    return X_train, X_test, y_train, y_test

def pipeline(scaler='StandardScaler',
            model_path='sklearn.linear_model',
            model_name='LinearRegression',
            params={}, classification=False):
    try:
        sc = getattr(
                __import__('sklearn.preprocessing', fromlist=[scaler]), scaler)
        imported_scaler = sc()
    except:
        return f"scaler:{scaler} inconnu"
    try:
        md = getattr(__import__(model_path, fromlist=[model_name]), model_name)
        import_model = md(**params)
    except:
        return f"model {model_name} ou {model_path} inconnu"
    
    if classification:
        return make_pipeline(imported_scaler, import_model)
    else:
        return make_pipeline(imported_scaler, MultiOutputRegressor(import_model))

def cross_validation_score(pipe, data, list_y, list_X, cv=5, classification=False):
    
    X_train, X_test, y_train, y_test = data_split(data, list_y, list_X, classification)
    cv_score = cross_val_score(pipe, X_train, y_train, cv)
    
    return cv_score, X_train, X_test, y_train, y_test

def get_train_pipeline(data, list_y, list_X, cv=5,
                        classification=False,
                        scaler='StandardScaler',
                        model_path='sklearn.linear_model', model_name='LinearRegression', params={}, return_full_set=False):
    pipe = pipeline(scaler, model_path, model_name, params, classification)
    cv_score, X_train, X_test, y_train, y_test = cross_validation_score(
                                                pipe, data, list_y, list_X, cv, classification)
    pipe.fit(X_train, y_train)
    
    if return_full_set:
        return pipe, cv_score, X_train, X_test, y_train, y_test
    else:
        return pipe, cv_score, X_test, y_test
# def custom_point_match(W=3, D=1, L=0)
#     def custom_point():
#     foot_metric = make_scorer(
#         lambda y_true, y_pred: mean_squared_log_error(y_true, y_pred)**0.5,
#         greater_is_better=False)
#     score_baseline = cross_val_score(pipe,
#                                      X_train,
#                                      y_train,
#                                      cv=5,
#                                      scoring=foot_metric).mean()









