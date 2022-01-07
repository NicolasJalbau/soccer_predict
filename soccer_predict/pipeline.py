# pylint: disable=missing-module-docstring

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
#from sklearn.metrics import make_scorer

def data_split(data, list_y, list_X, classification):
    """
    Paramètres obligatoires:

        data: dataset (pd.DataFrame)
        list_y: liste des colonnes à conserver pour y (list)
        list_X: liste des colonnes à DROP pour X (list)
        classification: type de tâche (classification=True/False)

    """
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
    """
    Paramètres optionnels:

        scaler: nom du scaler à utiliser dans le pipe (str, ex: "RobustScaler")
        model_path: chemin d'accès à la famille du modèle à utiliser dans le pipe (str, ex: "sklearn.neighbors")
        model_name: nom du modèle à utiliser (str, ex:"KNeighborsRegressor")
        params: dictionnaire contenant les hyperparamètres du modèle (dict, ex:{"n_neighbors": 5, "weights": 'uniform'})
        classification: type de tâche (classification=True/False)

        Si classification=True le modèle est inséré dans un MultiOutputRegressor
    """
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
    """
    Paramètres obligatoires:

        pipe: pipeline utilisé pour la cross validation(sklearn.pipeline.Pipeline)
        data: dataframe pour la cross validation(pd.DataFrame)
        list_y: liste des colonnes à conserver pour y (list)
        list_X: liste des colonnes à DROP pour X (list)
        Paramètres optionnels:
        cv: nombre de fold demandé pour la cross validation(int)
        classification: type de tâche (classification=True/False)

    """
    X_train, X_test, y_train, y_test = data_split(data, list_y, list_X, classification)
    cv_score = cross_val_score(pipe, X_train, y_train, cv=cv)
    
    return cv_score, X_train, X_test, y_train, y_test

def get_train_pipeline(data, list_y, list_X, cv=5,
                        classification=False,
                        scaler='StandardScaler',
                        model_path='sklearn.linear_model', model_name='LinearRegression', params={}, return_full_set=False):
    """
        Paramètres obligatoires:

            data: dataframe pour la cross validation(pd.DataFrame)
            list_y: liste des colonnes à conserver pour y (list)
            list_X: liste des colonnes à DROP pour X (list)
        
        Paramètres optionnels:
            cv: nombre de fold demandé pour la cross validation(int)
            classification: type de tâche (classification=True/False)
            scaler: nom du scaler à utiliser dans le pipe (str, ex: "RobustScaler")
            model_path: chemin d'accès à la famille du modèle à utiliser dans le pipe (str, ex: "sklearn.neighbors")
            model_name: nom du modèle à utiliser (str, ex:"KNeighborsRegressor")
            params: dictionnaire contenant les hyperparamètres du modèle (dict, ex:{"n_neighbors": 5, "weights": 'uniform'})
            return_full_set: indique si l'on souhaite récupérer uniquement le X_test/y_test (False) ou X_train/y_train et X_test/y_test(True)(bool)

    """
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


def grid_search(pipe, X_train, y_train,cv=5):
    """
    Paramètres obligatoires:

        pipe: pipe à grid search (pd.DataFrame)
        X_train: X pour entrainement (pd.DataFrame)
        y_train: y pour entrainement (pd.DataFrame)

    Paramètres optionnel:

        cv: nombre de fold demandé pour la cross validation(int)

    """
    params = pipe.get_params()
    for k, v in params.items():
        print(f"params: {k} -- {v}")
    
    grid_search_params = {}

    while True:   
        key = input("Enter the param you want to grid search or exit/N/n") 
        if key.lower() in ['exit', 'n', 'q'] : break;
        
        if params.get(key, False):
            value = input("Enter list of value to grid search (separator = , ):")
            value = [v.strip() for v in value.split(',') ]
            value = [True if v == 'True' else False if v=='False' else v for v in value ]                
            
            grid_search_params[key] = value
        else:
            print(f"{key} isn't a valid params name. Try again")
            
        
    grid_search = GridSearchCV(pipe,
                               param_grid=grid_search_params,
                               cv=cv,
                               n_jobs = -1
                               )

    grid_search.fit(X_train, y_train)
    for k,v in grid_search.best_params_.items():
        print("Result:")
        print(f"params: {k} -- {v}")
    print(grid_search.best_score_)
    
    return grid_search









