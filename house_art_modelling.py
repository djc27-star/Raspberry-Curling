# import modules
import pandas as pd
from pandas import isna
import numpy as np
# read in data 
data = pd.read_csv("house_art.csv")

# check for nas
non_values = isna(data).sum()/1460
non_values = non_values[non_values > 0]

#remove data with >20% missing values
data = data.drop(['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
          'BsmtFinType2','FireplaceQu','PoolQC','Fence','MiscFeature'],axis = 1)

# data description reveals values for some na values
data['GarageType'] = data['GarageType'].fillna('No Garage')
data['GarageFinish'] = data['GarageFinish'].fillna('No Garage')
data['GarageQual'] = data['GarageQual'].fillna('No Garage')
data['GarageCond'] = data['GarageCond'].fillna('No Garage')
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
data['GarageCars'] = data['GarageCars'].fillna(0)
data['GarageArea'] = data['GarageArea'].fillna(0)
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

# replace na with median in numerical data
for col in data.select_dtypes(include=np.number):
    data[col] = data[col].fillna(data[col].median())

# replace ns with mode in catagorical data
data.groupby('Electrical').size()
data['Electrical'] = data['Electrical'].fillna('SBrkr')


