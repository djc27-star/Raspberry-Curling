# import modules
import pandas as pd
from pandas import isna
import numpy as np
from pandas import get_dummies
from sklearn.preprocessing import StandardScaler

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


# remove response variable from data
target = data['SalePrice']
del data['SalePrice']



