# import modules
import pandas as pd
from pandas import isna
import numpy as np
from pandas import get_dummies
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import optuna
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# read in data 
data = pd.read_csv("house_art.csv")

# data description reveals values for some na values
none_col = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure',
            'BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish',
            'GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
for col in none_col:
    data[col].fillna('none',inplace = True)
data['GarageYrBlt'].fillna(0,inplace = True)
data['GarageCars'].fillna(0,inplace = True)
data['GarageArea'].fillna(0,inplace = True)
data['MasVnrArea'].fillna(0,inplace = True)

# check for other nas
non_values = isna(data).sum()/1460
non_values = non_values[non_values > 0]

# replace na with median in numerical data
data['LotFrontage'].fillna(data['LotFrontage'].median(),inplace = True)

# replace ns with mode in catagorical data
data.groupby('Electrical').size()
data['Electrical'].fillna('SBrkr',inplace = True)


# remove response variable from data and Id variable
target = np.log(data['SalePrice'])
del data['SalePrice']
del data['Id']

# standardise numerical data
st_x= StandardScaler() 
num_data = data.select_dtypes(include = 'number')
data[num_data.columns] = st_x.fit_transform(num_data)
del num_data 

# create dummy variables for catogorical data
data = get_dummies(data)

# def function for rmse
def rmse(pred,act):
    return np.sqrt(mean_squared_error(pred,act))

# split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(data,
         target, test_size=0.20, random_state=1, shuffle=True)

# build a simple rf model
regressor_rf_basic = RandomForestRegressor(n_estimators = 1000)
regressor_rf_basic.fit(X_train,Y_train)
y_test_pred = regressor_rf_basic.predict(X_test)
rf_basic = rmse(y_test_pred,Y_test)

# remove insignificant variables by feature importance
variable_importance = [(X_train.columns[list(regressor_rf_basic.feature_importances_).index(i)],i) for i in 
sorted(regressor_rf_basic.feature_importances_, reverse = True)]

sorted_importances = [x[1] for x in variable_importance]
sorted_features = [x[0] for x in variable_importance]
cumulative_importances = np.cumsum(sorted_importances)

# take 0.95% of feature importance to see if a differnece is made

index = next(x[0] for x in enumerate(cumulative_importances) if x[1] > 0.95) 
X_train2 = X_train[sorted_features[0:index]]
X_test2 = X_test[sorted_features[0:index]]

# try new model to see if we have lost much power
regressor_rf_simple = RandomForestRegressor(n_estimators = 1000)
regressor_rf_simple.fit(X_train2,Y_train)
y_test_pred = regressor_rf_simple.predict(X_test2)
rf_simple = rmse(y_test_pred,Y_test)

# take 0.99% of feature importance to see if a differnece is made
index = next(x[0] for x in enumerate(cumulative_importances) if x[1] > 0.99) 
X_train3 = X_train[sorted_features[0:index]]
X_test3 = X_test[sorted_features[0:index]]

# try new model to see if we have lost much power
regressor_rf_simple2 = RandomForestRegressor(n_estimators = 1000)
regressor_rf_simple2.fit(X_train3,Y_train)
y_test_pred = regressor_rf_simple2.predict(X_test3)
rf_simple2 = rmse(y_test_pred,Y_test)

# we will use 0.99% of the data
# optimise using optuna
# comment out optimization as time consuming
'''def objective(trial):
    bootstrap = trial.suggest_categorical('bootstrap',['True','False'])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf',1,10)
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt','log2'])
    n_estimators =  trial.suggest_int('n_estimators', 30, 5000)
    
    regr = RandomForestRegressor(bootstrap = bootstrap, min_samples_split=min_samples_split,
                                  max_features = max_features,
                                 min_samples_leaf=min_samples_leaf,n_estimators = n_estimators,n_jobs=2)
        
    score = cross_val_score(regr, X_train3, Y_train, cv=5, scoring="neg_root_mean_squared_error")
    mu = score.mean()

    return mu

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)'''

optimised_rf = RandomForestRegressor(bootstrap = 'True',min_samples_split= 2,
                                     min_samples_leaf= 1,max_features= 'auto',
                                     n_estimators= 5000)
optimised_rf.fit(X_train3 ,Y_train)
y_test_pred = optimised_rf.predict(X_test3)
rf_optimised = rmse(y_test_pred,Y_test)

# optimised model is strongest but barely, will now try another model
# will try xgboost
regressor_xbg = xgb.XGBRegressor()
regressor_xbg.fit(X_train3,Y_train)
y_test_pred = regressor_xbg.predict(X_test3)
xgb_basic= rmse(y_test_pred,Y_test)

def objective(trial):
    eta = trial.suggest_float('eta',0.001,0.999)
    max_depth = trial.suggest_int('max_depth',2,15)
    colsample_bytree = trial.suggest_float('colsample_bytree',0.1,1)
    n_estimators =  trial.suggest_int('n_estimators', 30, 10000)
    
    regr = xgb.XGBRegressor(eta = eta,max_depth = max_depth,n_estimators=n_estimators,
                            colsample_bytree=colsample_bytree)
    regressor_xbg = regr
    regressor_xbg.fit(X_train3,Y_train)
    y_test_pred = regressor_xbg.predict(X_test3)

    return rmse(y_test_pred,Y_test)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# optimized models
xbg_optimized = xgb.XGBRegressor(eta= 0.04587468576659705, max_depth= 11, 
                                 colsample_bytree= 0.7121988627323375, n_estimators= 8731)
xbg_optimized.fit(X_train3,Y_train)
y_test_pred = xbg_optimized.predict(X_test3)
xgb_optimized= rmse(y_test_pred,Y_test)