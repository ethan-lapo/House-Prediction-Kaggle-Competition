import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import math
from pandas.api.types import is_numeric_dtype
df = pd.read_csv('C:/Users/Ethan Lapaczonek/Downloads/house-prices-advanced-regression-techniques/train.csv')

print(df)
print(df.info())
#Overall goal is to predict what a house sells for

print(df.describe())
print(df.dtypes)
#Encode string columns
from sklearn.preprocessing import LabelEncoder
df = df.drop("Id", axis="columns")
print(df["SalePrice"])
label_encoder = LabelEncoder()
x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical = df.select_dtypes(exclude=['object'])
df = x_numerical.join(x_categorical)
print(df)
print(df["SalePrice"])

pd.set_option('display.max_rows', 500)

nan_count = df.isna().sum()
print(nan_count)
#Fix Lot Footage and Garage Year Built NA's
median = df.median()
df.fillna(median, inplace=True)
nan_count = df.isna().sum()
print(nan_count)


X = df.drop(['SalePrice'], axis=1)
y = df['SalePrice']
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

rf = RandomForestRegressor(n_estimators=50, random_state=42)
model = rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(y_pred)
print(rf.score(X_test, y_test))
mse = mean_squared_error(y_test.values.ravel(), y_pred)
r2 = r2_score(y_test.values.ravel(), y_pred)
print('Mean Squared Error:', round(mse, 2))
print('R-squared scores:', round(r2, 2))


#Test it with Kaggle test set
test_data = pd.read_csv('C:/Users/Ethan Lapaczonek/Downloads/house-prices-advanced-regression-techniques/test.csv')
ids = test_data.pop("Id")
label_encoder = LabelEncoder()
x_categorical_test = test_data.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical_test = test_data.select_dtypes(exclude=['object'])
test_df = x_numerical_test.join(x_categorical_test)
nan_count_test = test_df.isna().sum()
mean_test = test_df.mean()
test_df.fillna(mean_test, inplace=True)
predictions = rf.predict(test_df)
output = pd.DataFrame({"Id":ids, "SalePrice":predictions.squeeze()})

sample_submission_df = pd.read_csv('C:/Users/Ethan Lapaczonek/Downloads/house-prices-advanced-regression-techniques/sample_submission.csv')
sample_submission_df['SalePrice'] = rf.predict(test_df)
sample_submission_df.to_csv('C:/Users/Ethan Lapaczonek/Downloads/house-prices-advanced-regression-techniques/submission.csv', index=False)
print(sample_submission_df)
#Got a final score of .15

#Fine tuning
#normalizing the skewness of price
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor

regr_trans = TransformedTargetRegressor(regressor=rf, transformer=QuantileTransformer(output_distribution='normal'))
regr_trans.fit(X_train, y_train)
yhat = regr_trans.predict(X_test)
print(f"R^2 is: {r2_score(y_test, yhat)}, MSE is: {mean_squared_error(y_test,yhat)}")
#Test it with Kaggle test set
test_data = pd.read_csv('C:/Users/Ethan Lapaczonek/Downloads/house-prices-advanced-regression-techniques/test.csv')
ids = test_data.pop("Id")
label_encoder = LabelEncoder()
x_categorical_test = test_data.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical_test = test_data.select_dtypes(exclude=['object'])
test_df = x_numerical_test.join(x_categorical_test)
nan_count_test = test_df.isna().sum()
median_test = test_df.median()
test_df.fillna(median_test, inplace=True)
predictions = regr_trans.predict(test_df)
output = pd.DataFrame({"Id":ids, "SalePrice":predictions.squeeze()})

sample_submission_df = pd.read_csv('C:/Users/Ethan Lapaczonek/Downloads/house-prices-advanced-regression-techniques/sample_submission.csv')
sample_submission_df['SalePrice'] = regr_trans.predict(test_df)
sample_submission_df.to_csv('C:/Users/Ethan Lapaczonek/Downloads/house-prices-advanced-regression-techniques/submission.csv', index=False)
print(sample_submission_df)


from sklearn.ensemble import GradientBoostingRegressor
model_gb = GradientBoostingRegressor(n_estimators=1000, random_state=42, max_depth=2)
model_gb = model_gb.fit(X_train, y_train)
pred_gb = model_gb.predict(X_test)
mse_gb = mean_squared_error(y_test, pred_gb)
print(f"GRADIENT BOOSTED RANDOM FOREST REGRESSION IS : {mse_gb}")
r2_gb = r2_score(y_test, pred_gb)
print(f"GRADIENT R^2 IS: {r2_gb}")
sample_submission_df['SalePrice'] = model_gb.predict(test_df)
sample_submission_df.to_csv('C:/Users/Ethan Lapaczonek/Downloads/house-prices-advanced-regression-techniques/submission(gb).csv', index=False)

print(list(X_train.columns))


MSSubClass, LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, firstFlrSF, secondFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageYrBlt, GarageCars, GarageArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, threeSsnPorch, ScreenPorch, PoolArea,  MiscVal, MoSold, YrSold, MSZoning, Street, Alley, LotShape, LandContour,  Utilities, LotConfig, LandSlope, Neighborhood,Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir, Electrical, KitchenQual,Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive, PoolQC, Fence, MiscFeature, SaleType, SaleCondition = input('Enter MSSubClass: '), input('\n LotFrontage: '), input('\n LotArea: '), input('\n OverallQual: '), input('\n OverallCond: '), input('\n YearBuilt: '), input('\n YearRemodAdd: '), input('\n MasVnrArea: '), input('\n BsmtFinSF1: '), input('\n BsmtFinSF2: '), input('\n BsmtUnfSF: '), input('\n TotalBsmtSF: '), input('\n 1stFlrSF: '), input('\n 2ndFlrSF: '), input('\n LowQualFinSF: '), input('\n GrLivArea: '), input('\n BsmtFullBath: '), input('\n BsmtHalfBath: '), input('\n FullBath: '), input('\n HalfBath: '), input('\n BedroomAbvGr: '), input('\n KitchenAbvGr: '), input('\n TotRmsAbvGrd: '), input('\n Fireplaces: '), input('\n GarageYrBlt: '), input('\n GarageCars: '), input('\n GarageArea: '), input('\n WoodDeckSF: '), input('\n OpenPorchSF: '), input('\n EnclosedPorch: '), input('\n 3SsnPorch: '), input('\n ScreenPorch: '), input('\n PoolArea: '), input('\n MiscVal: '), input('\n MoSold: '), input('\n YrSold: '), input('\n MSZoning: '), input('\n Street: '), input('\n Alley: '), input('\n LotShape: '), input('\n LandContour: '), input('\n Utilities: '), input('\n LotConfig: '), input('\n LandSlope: '), input('\n Neighborhood '), input('\n Condition1: '), input('\n Condition2: '), input('\n BldgType: '), input('\n HouseStyle: '), input('\n RoofStyle: '), input('\n RoofMatl: '), input('\n Exterior1st: '), input('\n Exterior2nd: '), input('\n MasVnrType: '), input('\n ExterQual: '), input('\n ExterCond: '), input('\n Foundation: '), input('\n BsmtQual: '), input('\n BsmtCond: '), input('\n BsmtExposure: '), input('\n BsmtFinType1: '), input('\n BsmtFinType2: '), input('\n Heating: '), input('\n HeatingQC: '), input('\n CentralAir: '), input('\n Electrical: '), input('\n KitchenQual: '), input('\n Functional: '), input('\n FireplaceQu: '), input('\n GarageType: '), input('\n GarageFinish: '), input('\n GarageQual: '), input('\n GarageCond: '), input('\n PavedDrive: '), input('\n PoolQC: '), input('\n Fence: '), input('\n MiscFeature: '), input('\n SaleType: '), input('\n SaleCondition: ')
X_user_input = pd.DataFrame(columns=['MSSubClass','LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'firstFlrSF', 'secondFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'threeSsnPorch', 'ScreenPorch', 'PoolArea',  'MiscVal', 'MoSold', 'YrSold', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',  'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood','Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual','Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'])
label_encoder = LabelEncoder()
x_categorical_user = X_user_input.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical_user = X_user_input.select_dtypes(exclude=['object'])
X_user_input = x_numerical_user.join(x_categorical_user)
X_user_input.fillna(median, inplace=True)
user_predict = model_gb.predict(X_user_input)
print(f"The predicted sales price of the house is: {user_predict}")
user_accuracy = model_gb.score(X_user_input, y_test)
print(f"The accuracy of this model is: {user_accuracy}")
#Make model better by decreasing MSE and fine tune parameters

'''
characteristics = X.columns
importances = list(rf.feature_importances_)
characteristics_importances = [(characteristic, round(importance, 2)) for characteristic, importance in zip(characteristics, importances)]
characteristics_importances = sorted(characteristics_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in characteristics_importances]

#take only the top important variables (OverallQual, GrLivArea, BsmtFinSF1, TotalBsmtSF, 1stFlrSF, 2ndFlrSF):
X_new = df[["OverallQual", "GrLivArea", "BsmtFinSF1"]]
print(X_new)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
rf_new = RandomForestRegressor(n_estimators=1000)
model_new = rf_new.fit(X_train, y_train)
y_pred_new = rf_new.predict(X_test)
print(y_pred_new)
print(rf_new.score(X_test, y_test))
mse_new = mean_squared_error(y_test.values.ravel(), y_pred_new)
r2_new = r2_score(y_test.values.ravel(), y_pred_new)
print('OLD MSE IS:', round(mse, 2))
print('Mean Squared Error NEW MODEL IS :', round(mse_new, 2))
print('R-squared scores:', round(r2_new, 2))
predictions_new = rf_new.predict(test_df[["OverallQual", "GrLivArea", "BsmtFinSF1" ]])
output_new = pd.DataFrame({"Id":ids, "SalePrice":predictions_new.squeeze()})
print(output_new)

sample_submission_df_new = pd.read_csv('C:/Users/Ethan Lapaczonek/Downloads/house-prices-advanced-regression-techniques/sample_submission.csv')
sample_submission_df_new['SalePrice'] = rf_new.predict(test_df[["OverallQual", "GrLivArea", "BsmtFinSF1"]])
sample_submission_df_new.to_csv('C:/Users/Ethan Lapaczonek/Downloads/house-prices-advanced-regression-techniques/submission.csv', index=False)
print(sample_submission_df_new)
#Worse Score (gave me 0.17355)
'''

'''
#Correlation between variables
print(list(df.columns))
df_float = df.select_dtypes(include=[np.float64])
df_int = df.select_dtypes(include=[np.int64])
df_num = pd.concat([df_int, df_float], axis=1)
print(df_num)
corr = (df_num.corr())
#Regression model
df_num_nona = df_num.dropna()
Y = df_num_nona["SalePrice"]
print(Y)

X = df_num_nona.drop("SalePrice", axis="columns")
print(X)
import statsmodels.api as sm

model_reg = LinearRegression()
model_reg.fit(X,Y)
print(model_reg.coef_)
print(model_reg.score(X,Y))
x = sm.add_constant(X)
model = sm.OLS(Y, x).fit()
print(model.summary())
#Check corr for high P-value variables and drop variables

#Backward Elimination
x_back = x.drop(["Id", "BsmtFinSF2", "OpenPorchSF", "EnclosedPorch", "LowQualFinSF", "HalfBath", "BsmtUnfSF", "1stFlrSF", "2ndFlrSF", "GarageCars", "BsmtHalfBath", "3SsnPorch", "GarageYrBlt", "MoSold", "LotFrontage", "MiscVal", "YrSold","TotalBsmtSF", "Fireplaces","WoodDeckSF", "YearRemodAdd", "GrLivArea", "PoolArea"], axis="columns")
model = sm.OLS(Y,x_back).fit()
print(model.summary())

#Forward Selection
x_for = x[["const", "OverallQual", "MSSubClass", "GarageCars", "MasVnrArea", "BsmtFinSF1", "LotArea", "FullBath", "ScreenPorch"]]
model = sm.OLS(Y,x_for).fit()
print(model.summary())
print(x_for.corr())

df = df.drop("Id", axis="columns")
print(df)
df_object = (df.select_dtypes(include=[np.object_]))
print(df_object)
#Convert all these columns to numbers
one_hot_encoded_data = pd.get_dummies(df_object) 
print(one_hot_encoded_data)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

print(df_num)
new_df = one_hot_encoded_data.join(df_num)
print(new_df[new_df.isna().any(axis=1)])
new_df = new_df.dropna()
print(new_df.shape)
accuracy = accuracy_score(test_y, y_pred)
print("Accuracy:", accuracy)
print(classification_report(test_y, y_pred))

model_gradient = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=100, max_features=5 )
model_gradient.fit(train_X,train_y)
print("WORKING")
pred_y = model_gradient.predict(test_X)
acc = accuracy_score(test_y, pred_y)
print(f"Accuracy is : {acc}")

print(classification_report(test_y,model_gradient.predict(test_X)))
'''
