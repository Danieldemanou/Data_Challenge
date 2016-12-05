
# coding: utf-8

# # Prédiction de l'obtention d'un brevet

# SD210 - Base de l'apprentissage statistique - Challenge -
# DEMANOU WAMO

import numpy as np
import pandas as pd


from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sklearn.cross_validation as cross_validation
from sklearn.grid_search import GridSearchCV
train_f = './train.csv'
test_f = './test.csv'
df = pd.read_csv(train_f, sep=';')
df_test = pd.read_csv(test_f, sep=';')

X_test =  np.zeros(((df_test[['APP_NB']].values).shape[0]))
X_train = np.zeros(((df[['APP_NB']].values).shape[0]))

y_train = df.VARIABLE_CIBLE == 'GRANTED'

for i in ['VOIE_DEPOT','SOURCE_IDX_RAD', 'COUNTRY', 'SOURCE_BEGIN_MONTH', 'FISRT_APP_COUNTRY', 'FISRT_APP_TYPE',
            'LANGUAGE_OF_FILLING', 'TECHNOLOGIE_SECTOR', 'TECHNOLOGIE_FIELD',
            'FISRT_INV_COUNTRY', 'FISRT_INV_TYPE', 'SOURCE_CITED_AGE', 'SOURCE_IDX_ORI'] :
    le = LabelEncoder()
    le.fit(np.hstack((df[i].values,df_test[i].values)))
    X_train = np.c_[le.transform(df[i].values),X_train]
    X_test = np.c_[le.transform(df_test[i].values),X_test]
    
encoder=OneHotEncoder(sparse=False)
encoder.fit(np.vstack((X_train,X_test)))
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

feature_chif = []
for feature in df.columns :
    if type(df[feature][0]) is not str :
        feature_chif.append(feature)


for i in ['FIRST_CLASSE','MAIN_IPC'] :
    le = LabelEncoder()
    le.fit(np.hstack((df[i].values,df_test[i].values)))
    X_train = np.c_[le.transform(df[i].values),X_train]
    X_test = np.c_[le.transform(df_test[i].values),X_test]
    
    
X_train = np.c_[df[feature_chif].values,X_train]
X_test = np.c_[df_test[feature_chif].values,X_test]

d_test=df_test['BEGIN_MONTH'].str.split('/', expand=True, n=1).astype(float)
d_train=df['BEGIN_MONTH'].str.split('/', expand=True, n=1).astype(float)
X_train = np.c_[(d_train.values)[:,0]+((d_train.values)[:,1]-1988)*12,X_train]
X_test = np.c_[(d_test.values)[:,0]+((d_test.values)[:,1]-1988)*12,X_test]

for i in [ 'FILING_MONTH','PRIORITY_MONTH', 'PUBLICATION_MONTH'] :
        t_test=df_test[i].str.split('/', expand=True, n=1).astype(float)
        t_train=df[i].str.split('/', expand=True, n=1).astype(float)
        X_train = np.c_[((t_train.values)[:,0]+((t_train.values)[:,1]-1988)*12)- ((d_train.values)[:,0]+((d_train.values)[:,1]-1988)*12),X_train]
        X_test = np.c_[((t_test.values)[:,0]+((t_test.values)[:,1]-1988)*12)- ((d_test.values)[:,0]+((d_test.values)[:,1]-1988)*12),X_test]


imputer = Imputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


clf = GradientBoostingClassifier()
param_grid = dict(n_estimators=[800],max_depth = [8], max_features=[0.3],learning_rate = [0.1],min_samples_split = [600],min_samples_leaf = [40],subsample = [1.],random_state = [1])
grid = GridSearchCV(clf, param_grid = param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)
print("Best score %f" %grid.best_score_)
#score = cross_validation.cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')

Y_predict = grid.predict_proba(X_test)
np.savetxt('y_predDaniel.txt', Y_predict, fmt='%s')

