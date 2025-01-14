# Import Libraries
import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

#  Load the Dataset
data_set = pd.read_csv(r"C:\SMEC\Data_Science\Project\DATASET\Salary_Data_ML.csv")
data_set.dropna(inplace=True)
# print(data_set)

df1=data_set.select_dtypes(exclude=['object'])
for coloumn in df1:
    plt.figure(figsize=(17,1))
    sns.boxplot(data=df1,x=coloumn)
# print(df1)
# plt.show()

# x=data_set.iloc[:,3:5].values
# y=data_set["Salary"].values
# x=pd.DataFrame(x)
# y=pd.DataFrame(y)
plt.scatter(data_set['Years of Experience'],data_set['Salary'])
# plt.show()

# Droping the unwanted coloum

x=data_set.drop(columns=["Age","Gender","Education Level",'Salary'],axis=1)
y=data_set['Salary']
# print(x)
# print(y)
column_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = column_transformer.fit_transform(x)

# sns.scatterplot(x="Job Title",y="Salary",data=data_set)
# plt.show()

# Split dataset into training set and test set

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print(x_test)
print(x_train)
print(y_test)
print(y_train)

model=LinearRegression()
model.fit(x_train,y_train)

# Boosting the r2 score

y_pred=model.predict(x_test)
print(y_pred)
accuracy1=r2_score(y_pred,y_test)
print(accuracy1)
model2=XGBRegressor()
model2.fit(x_train,y_train)
y_pred=model2.predict(x_test)
accuracy2=r2_score(y_test,y_pred)
print(accuracy2)

model3=GradientBoostingRegressor()
model3.fit(x_train,y_train)
y_pred4=model3.predict(x_test)
accuracy3=r2_score(y_pred,y_test)
print(accuracy3)



