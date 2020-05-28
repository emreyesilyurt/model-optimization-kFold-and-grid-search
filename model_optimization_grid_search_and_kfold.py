import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
train = pd.read_csv('/dataset/train.csv')
test = pd.read_csv('/dataset/test.csv')
gender = pd.read_csv('gender_submission.csv')
#test = test.iloc[:,1:]
#test = pd.concat([gender, test], axis = 1)


#data = pd.concat([train, test], axis = 0)
#columnsData = data.columns
#data = data.values
#data = pd.DataFrame(data = data, index = range(len(data)), columns = columnsData)
print(train.isnull().sum())

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')    
#data.iloc[:,5:6] = imputer.fit_transform(data.iloc[:,5:6])
train.iloc[:,5:6] = imputer.fit_transform(train.iloc[:,5:6])


imputer2 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
test.iloc[:,4:5] = imputer2.fit_transform(test.iloc[:,4:5])

imputer3 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
test[['Fare']] = imputer3.fit_transform(test[['Fare']])
print(train.isnull().sum())
print(test.isnull().sum())


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
sexTrain = train.iloc[:,4:5]
sexTrain = ohe.fit_transform(sexTrain).toarray()
sexTrain = pd.DataFrame(data = sexTrain, index = range(len(sexTrain)), columns = ['Female', 'Male'])
#sex = sex.values

ohe2 = OneHotEncoder()
sexTest = test.iloc[:,3:4]
sexTest = ohe.fit_transform(sexTest).toarray()
sexTest = pd.DataFrame(data = sexTest, index = range(len(sexTest)), columns = ['Female', 'Male'])

#x_train = pd.concat([train['Pclass'], train['Age']], axis = 1)
x_train = pd.concat([train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']], sexTrain['Female']], axis = 1)

x_test = pd.concat([test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']], sexTest['Female']], axis = 1)

#y_train = train.iloc[:,1:2]
y_train = pd.concat([train['Survived']])

y_test = gender.iloc[:,1:]

"""
X = pd.concat([train.iloc[:,2:3], sex.iloc[:,0:1], data.iloc[:,5:8], data.iloc[:,9:10]], axis = 1)
Y = data.iloc[:,1:2]
X = X.astype(float)
Y = Y.astype(int)
x = X.values
y = Y.values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)
"""
from xgboost import XGBClassifier

xgb = XGBClassifier(base_score=0.8, colsample_bylevel = 1, learning_rate=0.1, missing=False, n_estimators=500,
       objective='binary:logistic', reg_alpha=3, reg_lambda=13, subsample = 1, gamma = 0)

xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)

print('y_test', y_test.shape, '\ny_pred_xgb', y_pred_xgb.shape)
cm = confusion_matrix(y_test, y_pred_xgb)
print('XGB CM\n', cm)

"""

params = [{'learning_rate':[0.1,0.01],
           'colsample_bytree':[1,3,5,9],
           'gamma':[0,1,2,7],
           'reg_alpha':[3,4,8],
           'reg_lambda':[1,13,15,32],
           'n_estimators':[200,500],
           'colsample_bylevel':[1,2,3,8],
           'missing':[False, True],
           'subsample':[1,2,3],
           'base_score':[0.2,0.5,0.8]
           }
    ]
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(estimator = xgb,
                  param_grid = params,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)

grid_search = gs.fit(x_train, y_train)
best_result = grid_search.best_score_
best_params = grid_search.best_params_
print('Best_Result', best_result)
print('Best_Params', best_params)

"""
params = [{'learning_rate':[0.1,0.01],
           'colsample_bytree':[1,3],
           'gamma':[0,1],
           'reg_alpha':[3,4,8],
           'reg_lambda':[1,13,15],
           'n_estimators':[200,500],
           'missing':[False, True],
           'subsample':[1,2],
           'base_score':[0.2,0.8]
           }
]

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state = 0)
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('LogisticRegression CM\n', cm)

from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
#svc = SVC(kernel = 'poly')
#svc = SVC(kernel = 'rbf')
#svc = SVC(kernel = 'precomputed')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('SVC CM\n', cm)
"""
params = [{
    'kernel' : ['linear', 'rbf', 'poly']
    }]

gs = GridSearchCV(
    estimator = svc,
    param_grid = params,
    scoring = 'accuracy',
    cv = 10,
    n_jobs = -1
    )

grid_search = gs.fit(x_train, y_train)
best_result = grid_search.best_score_
best_params = grid_search.best_params_
print('SVC Grid Search')
print('Best_Result', best_result)
print('Best_Params', best_params)
"""

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('GaussianNB CM\n', cm)


from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred = mnb.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('MultinomialNB CM\n', cm)


from sklearn.naive_bayes import ComplementNB
cnb = ComplementNB()
cnb.fit(x_train, y_train)
y_pred = cnb.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('ComplementNB CM\n', cm)

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(x_train, y_train)
y_pred = bnb.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('BernoulliNB CM\n', cm)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('KNeighborsClassifier CM\n', cm)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('RandomForestClassifier CM\n', cm)

"""
params = [
    {'n_estimators' : [5,10,15,20], 'criterion' : ['gini', 'entropy']}
    ]

gs = GridSearchCV(
    estimator = rfc,
    param_grid = params,
    scoring = 'accuracy',
    cv = 10,
    n_jobs = -1
    )

grid_search = gs.fit(x_train, y_train)
best_result = grid_search.best_score_
best_params = grid_search.best_params_
print('RFC Grid Search')
print('Best_Result', best_result)
print('Best_Params', best_params)
"""

#cross val score
crossVal= cross_val_score(estimator = xgb, X = x_train, y = y_train, cv = 4)
print('XGB Accuracy: ', crossVal.mean())
print('XGB Std: ', crossVal.std())


crossVal = cross_val_score(estimator = log_reg, X = x_train, y = y_train, cv = 4)
print('Logistic Regression Accuracy: ', crossVal.mean())
print('Logistic Regression Std: ', crossVal.std())

crossVal = cross_val_score(estimator = svc, X = x_train, y = y_train, cv = 4)
print('SVC Accuracy: ', crossVal.mean())
print('SVC Std: ', crossVal.std())

crossVal = cross_val_score(estimator = gnb, X = x_train, y = y_train, cv = 4)
print('GaussianNB Accuracy: ', crossVal.mean())
print('GaussianNB Std: ', crossVal.std())

crossVal = cross_val_score(estimator = mnb, X = x_train, y = y_train, cv = 4)
print('MultinomialNB Accuracy: ', crossVal.mean())
print('MultinomialNB Std: ', crossVal.std())

crossVal = cross_val_score(estimator = cnb, X = x_train, y = y_train, cv = 4)
print('ComplementNB Accuracy: ', crossVal.mean())
print('ComplementNB Std: ', crossVal.std())

crossVal = cross_val_score(estimator = bnb, X = x_train, y = y_train, cv = 4)
print('BernoulliNB Accuracy: ', crossVal.mean())
print('BernoulliNB Std: ', crossVal.std())


crossVal = cross_val_score(estimator = knn, X = x_train, y = y_train, cv = 4)
print('KNeighborsClassifier Accuracy: ', crossVal.mean())
print('KNeighborsClassifier Std: ', crossVal.std())

crossVal = cross_val_score(estimator = rfc, X = x_train, y = y_train, cv = 4)
print('RandomForestClassifier Accuracy: ', crossVal.mean())
print('RandomForestClassifier Std: ', crossVal.std())

passengerId = gender.iloc[:,0:1]
y_pred_xgb = pd.DataFrame(data = y_pred_xgb, index = range(len(y_pred_xgb)))
submit = pd.concat([passengerId, y_pred_xgb], axis = 1)

submit.to_csv(r'submission.csv', index=False)