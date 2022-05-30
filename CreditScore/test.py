import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('dataset/japan.csv')
test = pd.read_csv('dataset/test.csv')
train = train.replace('?',np.NaN)
train = train.replace(np.NaN, 0)
#print(train)
le = preprocessing.LabelEncoder()
train['Gender'] = le.fit_transform(train['Gender'])
train['Married'] = le.fit_transform(train['Married'])
train['Bank_Customer'] = le.fit_transform(train['Bank_Customer'])
train['Education'] = le.fit_transform(train['Education'])
train['Ethnicity'] = le.fit_transform(train['Ethnicity'])
train['Years_Employed'] = le.fit_transform(train['Years_Employed'])
train['Prior_Default'] = le.fit_transform(train['Prior_Default'])
train['Credit_Score'] = le.fit_transform(train['Credit_Score'])
train['Drivers_License'] = le.fit_transform(train['Drivers_License'])
train['Approved'] = le.fit_transform(train['Approved'])

test['f0'] = le.fit_transform(test['f0'])
test['f3'] = le.fit_transform(test['f3'])
test['f4'] = le.fit_transform(test['f4'])
test['f5'] = le.fit_transform(test['f5'])
test['f6'] = le.fit_transform(test['f6'])
test['f8'] = le.fit_transform(test['f8'])
test['f9'] = le.fit_transform(test['f9'])
test['f11'] = le.fit_transform(test['f11'])
test['f12'] = le.fit_transform(test['f12'])


cols = train.shape[1]
X = train.values[:, 0:cols-1] 
Y = train.values[:, cols-1]
Y = Y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
cols = train.shape[1]
test1 = test.values[:, 0:cols]
print(test1)

print(X.shape)
print(test1.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

def prediction(X_test, cls):
    y_pred = cls.predict(X_test)
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test,y_pred)*100
    return accuracy  

sm = SMOTE(random_state = 2) 
X_train,y_train = sm.fit_resample(X_train,y_train)
print("After OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train == 0))) 

print(X_train.shape)
print(test1.shape)
gbm_param_grid = {
     'colsample_bytree': np.linspace(0.5, 0.9, 5),
     'n_estimators':[100, 200],
     'max_depth': [10, 15, 20, 25]
}
gbm = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)
grid_mse = GridSearchCV(estimator = gbm, param_grid = gbm_param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose = 0)
gbm.fit(X_train, y_train)
pred = gbm.predict(X_test)
prediction_data = prediction(X_test, gbm) 
acc = cal_accuracy(y_test, prediction_data)
print(acc)
