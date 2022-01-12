import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn import set_config; set_config(display='diagram')
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

def data_split(data, list_y, list_X, classification):
    """
    Paramètres obligatoires:

        data: dataset (pd.DataFrame)
        list_y: liste des colonnes à conserver pour y (list)
        list_X: liste des colonnes à DROP pour X (list)
        classification: type de tâche (classification=True/False)

        return X_train, X_test, y_train, y_test
    """
    X = data.drop(columns=list_X)
    y = pd.DataFrame(data[list_y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if classification:
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

    return X_train, X_test, y_train, y_test

def cross_validation_score(pipe, data, list_y, list_X, classification=False, cv=5):
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
    if classification:
        cv_score = cross_val_score(pipe, X_train, y_train, cv=cv,scoring='accuracy')
    else:
        cv_score = cross_val_score(pipe, X_train, y_train, cv=cv,scoring=make_scorer(custom_score))

    return {'cv_score':cv_score, 'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}

class Posttreat(BaseEstimator, ClassifierMixin):
    def __init__(self, indice_tole =0.1):
        self.indice_tole = indice_tole

    def fit(self,X, y):
        return self
    def predict(self, X, y=None):
        # Ajouter la logique de post-traitement ici
        y_pred = pd.DataFrame(X)
        # création des colonnes de résultat
        y_pred['res'] = y_pred[0]-y_pred[1]
        # conversion des résultats pour calcul du score
        y_pred_new = y_pred['res'].apply(lambda x: 3 if x > self.indice_tole else 0 if x < -self.indice_tole else 1)

        return (y_pred_new.reset_index(drop=True))

def custom_score(y_t, y_p, indice_tol=0.1):
    """
    Paramètres obligatoires:

        y_t: np.array du y_test (np.ndarray)
        y_p: np.array du y_pred (np.ndarray)

    Paramètres optionnel:

        indice_tol: Indice de tolérance: différence déterminant l'intervalle comprenant les matchs nuls dans la prédiction.(float)
                    Exemple: indice_tol=0.1 -> Tout les matchs ayant une différence de buts entre [0.1|-0.1] sont un résultat nul
    """

    # réassignation des argument en pd.DataFrame
    y_pred = pd.DataFrame(y_p)
    y_test = pd.DataFrame(y_t)
    # création des colonnes de résultat
    y_test['res'] = y_test['homeGoals'] - y_test['awayGoals']
    # conversion des résultats pour calcul du score
    y_test['res'] = y_test['res'].apply(lambda x: 3 if x > indice_tol else 0 if x < -indice_tol else 1)
    y_test_new = pd.DataFrame(y_test['res'].reset_index(drop=True))
    # calcul de l'accuracy
    accuracy = (y_pred - y_test_new).value_counts()[0]\
    /(y_pred - y_test_new).value_counts().sum()

    return accuracy

class CustomMultiouputwrapper(MultiOutputRegressor):
    def transform(self, X, y=None):
        return self.predict(X)
    def fit_tranform(self, X, y):
        return self.fit(X, y).predict(X)


def pipeline(scaler='StandardScaler',
            model_path='sklearn.linear_model',
            model_name='LinearRegression', classification=False):
    """
    Paramètres optionnels:

        scaler: nom du scaler à utiliser dans le pipe (str, ex: "RobustScaler")
        model_path: chemin d'accès à la famille du modèle à utiliser dans le pipe (str, ex: "sklearn.neighbors")
        model_name: nom du modèle à utiliser (str, ex:"KNeighborsRegressor")
        classification: type de tâche (classification=True/False)

        Si classification=True le modèle est inséré dans un MultiOutputRegressor
    """
    try:
        sc = getattr(__import__('sklearn.preprocessing', fromlist=[scaler]), scaler)
    except:
        return f"scaler:{scaler} inconnu"

    try:
        md = getattr(__import__(model_path, fromlist=[model_name]), model_name)
    except:
        return f"model {model_name} ou {model_path} inconnu"

    if classification:
        return make_pipeline(sc(), md())
    else:
        return make_pipeline(sc(), CustomMultiouputwrapper(md()),Posttreat())


def pipeline_scalers(
                     model_path='sklearn.linear_model',
                     model_name='LinearRegression',
                     classification=False):
    """
    Paramètres optionnels:

        scaler: nom du scaler à utiliser dans le pipe (str, ex: "RobustScaler")
        model_path: chemin d'accès à la famille du modèle à utiliser dans le pipe (str, ex: "sklearn.neighbors")
        model_name: nom du modèle à utiliser (str, ex:"KNeighborsRegressor")
        classification: type de tâche (classification=True/False)

        Si classification=True le modèle est inséré dans un MultiOutputRegressor
    """
    try:
        sc = getattr(__import__('sklearn.preprocessing', fromlist=[scaler]),
                     scaler)
    except:
        return f"scaler:{scaler} inconnu"

    try:
        md = getattr(__import__(model_path, fromlist=[model_name]), model_name)
    except:
        return f"model {model_name} ou {model_path} inconnu"

    if classification:
        return make_pipeline(sc(), md())
    else:
        colRobustSc = [
            'Homeshots', 'Homedeep', 'Homeppda', 'Homefouls', 'HomeredCards',
            'Awaygoals', 'AwayxGoals', 'Awayshots', 'Awaydeep', 'Awayppda',
            'Awayfouls', 'AwayredCards'
        ]
        colStandSc = [
            'Homegoals', 'HomexGoals', 'HomeshotsOnTarget', 'Homecorners',
            'HomeyellowCards', 'AwayshotsOnTarget', 'Awaycorners',
            'AwayyellowCards'
        ]
        colMinMaxSc = [
            'homeOVA', 'homeATT', 'homeDEF', 'homeMID', 'awayOVA', 'awayATT',
            'awayDEF', 'awayMID'
        ]

        preproc = make_column_transformer((RobustScaler(), colRobustSc),
                                          (StandardScaler(), colStandSc),
                                          (MinMaxScaler(), colMinMaxSc),
                                          remainder='passthrough')
        return make_pipeline(preproc, CustomMultiouputwrapper(md()), Posttreat())


def get_train_pipeline(data, list_y, list_X, cv=5,
                        classification=False,
                        scaler='StandardScaler',
                        model_path='sklearn.linear_model', model_name='LinearRegression', return_full_set=False):
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
            return_full_set: indique si l'on souhaite récupérer uniquement le X_test/y_test (False) ou X_train/y_train et X_test/y_test(True)(bool)

    """
    # creation du pipeline
    pipe = pipeline(scaler, model_path, model_name, classification)
    # cross validation du pipeline
    temp_res = cross_validation_score(pipe, data, list_y, list_X,classification, cv=cv)
    # récupération du score et du dataset splitté
    cv_score = temp_res['cv_score']; X_train = temp_res['X_train']; X_test = temp_res['X_test']
    y_train = temp_res['y_train']; y_test = temp_res['y_test']

    # fit du pipeline
    pipe.fit(X_train, y_train)

    if return_full_set:
        # renvoi du pipeline et dataset complet
        return {'pipe':pipe, 'cv_score':cv_score,
                'X_train':X_train, 'X_test':X_test,
                'y_train':y_train, 'y_test':y_test}
    else:
        # renvoi du pipeline du split de test
        return {'pipe':pipe, 'cv_score':cv_score,
                'X_test':X_test,'y_test':y_test}


def grid_search(pipe, X_train, y_train,cv=5):
    """
    Paramètres obligatoires:

        pipe: pipe à grid search (sklearn.pipeline)
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
            temp = []
            for v in value:
                if isinstance(bool(v),bool) and v == 'True':
                    temp.append(True)
                elif isinstance(bool(v),bool) and v=='False':
                    temp.append(False)
                else:
                    try:
                        if isinstance(int(v),int):
                            temp.append(int(v))
                    except:
                        temp.append(v)
            value = temp

            grid_search_params[key] = value
        else:
            print(f"{key} isn't a valid params name. Try again")


    grid_search = GridSearchCV(pipe,
                               param_grid=grid_search_params,
                               cv=cv,
                               scoring=make_scorer(custom_score),
                               n_jobs = -1
                               )

    grid_search.fit(X_train, y_train)
    for k,v in grid_search.best_params_.items():
        print("Result:")
        print(f"params: {k} -- {v}")
    print(grid_search.best_score_)

    return grid_search
