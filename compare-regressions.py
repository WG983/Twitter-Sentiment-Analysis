import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

test_data = pd.read_csv(r'C:\Users\Kaktus\Desktop\python\science\price-prediction\test.csv')
train_data = pd.read_csv(r'C:\Users\Kaktus\Desktop\python\science\price-prediction\train.csv')

print(train_data.describe())

#Cleaning data
#filter and sort by null values

print(train_data.isnull().sum().sort_values(ascending = False).head(20))
print(test_data.isnull().sum().sort_values(ascending = False).head(20))

#remove columns with mostly null values

train_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis = 1, inplace = True)
test_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis = 1, inplace = True)

#replace missing values with 'NA' or values for specific variables

nulls = train_data.columns[train_data.isnull().any()].tolist()
print(nulls)
for index in nulls :
   train_data.loc[train_data[index].isnull() == True, index] = 'NA'
print(train_data.isnull().sum().sort_values(ascending = False).head(20))

nulls_test = test_data.columns[test_data.isnull().any()].tolist()
print(nulls_test)
nulls_test = [e for e in nulls_test if e not in ('MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GarageYrBlt', 'GarageCars', 'GarageArea')]
for index in nulls_test :
   test_data.loc[test_data[index].isnull() == True, index] = 'NA'
print(test_data.isnull().sum().sort_values(ascending = False).head(20))

print(test_data.MasVnrArea.value_counts())
sns.histplot(test_data['MasVnrArea'], kde = True, stat = 'density')

test_data.loc[test_data['MasVnrArea'].isnull() == True, 'MasVnrArea'] = test_data['MasVnrArea'].median()

print(test_data.BsmtFinSF1.value_counts())
sns.histplot(test_data['BsmtFinSF1'], kde = True, stat = 'density')
test_data.loc[test_data['BsmtFinSF1'].isnull() == True, 'BsmtFinSF1'] = 0

print(test_data.TotalBsmtSF.value_counts())
sns.histplot(test_data['TotalBsmtSF'], kde = True, stat = 'density')
test_data.loc[test_data['TotalBsmtSF'].isnull() == True, 'TotalBsmtSF'] = 0

print(test_data.GarageYrBlt.value_counts())
sns.histplot(test_data['GarageYrBlt'], kde = True, stat = 'density')
print(test_data.GarageYrBlt.max())
test_data.loc[test_data['GarageYrBlt'] == 2207, 'GarageYrBlt'] = 2007
sns.histplot(test_data['GarageYrBlt'], kde = True, stat = 'density')

test_data.loc[test_data['GarageYrBlt'].isnull() == True, 'GarageYrBlt'] = test_data['GarageYrBlt'].median()

print(test_data.GarageCars.value_counts())
sns.histplot(test_data['GarageCars'], kde = True, stat = 'density')
test_data.loc[test_data['GarageCars'].isnull()==True, 'GarageCars'] = test_data['GarageCars'].median()

print(test_data.GarageArea.value_counts())
sns.histplot(test_data['GarageArea'], kde = True, stat = 'density')
test_data.loc[test_data['GarageArea'].isnull()==True, 'GarageArea'] = 0

#encode object values with label encoder
le = LabelEncoder()
train_SaleCon = train_data['SaleCondition']
train_SaleCon_enc = le.fit_transform(train_SaleCon)

test_SaleCon = test_data['SaleCondition']
test_SaleCon_enc = le.fit_transform(test_SaleCon)

print(train_SaleCon_enc[0:100], test_SaleCon_enc[0:100])

#encode variables into binary numbers with get_dummies
train_data_bi = pd.get_dummies(train_data)
test_data_bi = pd.get_dummies(test_data)

print(train_data_bi.head())

#change data type into float
name_train = list(train_data_bi.select_dtypes(['int64']).columns)
name_test = list(test_data_bi.select_dtypes(['int64']).columns)

for i in name_train :
    train_data_bi[i] = train_data_bi[i].apply(float)
    
for i in name_test :
    test_data_bi[i] = test_data_bi[i].apply(float)

print(train_data_bi.info())
print(test_data_bi.info())

#scale features and split data
train_data_bi.dropna(axis = 0, subset = ['SalePrice'], inplace = True)

y = train_data_bi.SalePrice
x = train_data_bi.drop(['SalePrice'], axis = 1).select_dtypes(exclude = ['object'])

X_train, X_test, y_train, y_test = train_test_split(x.to_numpy(), y.to_numpy(), test_size = 0.25)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

decision_model = DecisionTreeRegressor()  
decision_model.fit(X_train_scaled, y_train) 
predicted_decision_trees = decision_model.predict(X_test_scaled)
print ("Mean Absolute Error using Decision Tree :", mean_absolute_error(y_test, predicted_decision_trees))
print("R-square coefficient using Decision Tree :", r2_score(y_test, predicted_decision_trees))

AdaBoost_model = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=300)
AdaBoost_model.fit(X_train_scaled, y_train)
predicted_AdaBoost = AdaBoost_model.predict(X_test_scaled)
print ("Mean Absolute Error using Boosted Decision Tree :", mean_absolute_error(y_test, predicted_AdaBoost))
print("R-square coefficient using Boosted Decision Tree :", r2_score(y_test, predicted_AdaBoost))

forest_model = RandomForestRegressor(n_estimators=100, max_depth=10)
forest_model.fit(X_train_scaled, y_train )
predicted_random_forest = forest_model.predict(X_test_scaled)
print("Mean Absolute Error using Random Forest:", mean_absolute_error(y_test, predicted_random_forest))
print("R-square coefficient using Random Forest :", r2_score(y_test, predicted_random_forest))

xg_model = XGBRegressor(n_estimators=100)
xg_model.fit(X_train_scaled, y_train)
predicted_XGBoost = xg_model.predict(X_test_scaled)
print("Mean Absolute Error using XGBoost :", mean_absolute_error(y_test, predicted_XGBoost))
print("R-square coefficient using XGBoost :", r2_score(y_test, predicted_XGBoost))
