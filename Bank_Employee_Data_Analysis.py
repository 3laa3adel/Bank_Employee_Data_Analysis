from numpy.core.numeric import correlate
from numpy.lib.function_base import median
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from pandas.core.frame import DataFrame
from scipy.stats import iqr
from scipy import stats
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import getpass
from sklearn.metrics import mean_squared_error, r2_score
data = {'ID': pd.Series(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']),
        'Name': pd.Series(['Ahmed', 'Ali', 'Bassem', 'Mohammed', 'Salma', 'Nada', 'Fatma', 'Hossam', 'Mazen', 'Abdullah', 'Rawan', 'Youssef', 'Maged', 'Menna', 'Salah']),
        'Age': pd.Series([25, 26, 25, 23, 30, 29, 23, 34, 40, 30, 51, 46, 32, 22, 26]),
        'Salary': pd.Series([5000, 10000, 4000, 12000, 7000, 5500, 14000, 3900, 17000, 8000, 11000, 4000, 5200, 12000, 7300]),
        'Rating': pd.Series([4, 3, 3, 2, 3, 4, 3, 3, 2, 4, 4, 3, 4, 3, 2])}
# Load the data
df = pd.DataFrame(data)
idx = ['1', '2', '3', '4', '5', '6', '7', '8',
       '9', '10', '11', '12', '13', '14', '15']
df.index = idx
print("################################################")
print("Bank Employee Data Analysis")
print("################################################")
while True:
    username = input("Enter Your Username : ")
    print("-----------------------------------")
    password = getpass.getpass(prompt='Enter Your Password: ', stream=None)
    print("-----------------------------------")
    if int(password) == 00000 and username == "aa_20":
        break
    else:
        print("-------------------------")
        print("Enter Your Data Correctly")
        print("-------------------------")
print(df)
print("###########################################")
print(df.info())
print("###########################################")
print("Mean Values in the Data")
print("Age      ", df.loc[:, 'Age'].mean())
print("Salary   ", df.loc[:, 'Salary'].mean())
print("Rating   ", df.loc[:, 'Rating'].mean())
print("dtype: float64")
print("###########################################")
print("Median Values in the Data")
print(df.median())
print("###########################################")
print("Mode Values in the Data")
print(df.mode())
print("###########################################")
print("Standard Deviation Values in the Data")
print(df.std())
print("###########################################")
print("Variance Values in the Data")
print(df.var())
print("###########################################")
print("Correlation Value in the Salary And Rating")
print(np.corrcoef(df.loc[:, 'Salary'], df.loc[:, 'Rating']))
print("###########################################")
print("Regression Values in the Salary And Rating")
X = df.loc[:, 'Rating']
Y = df.loc[:, 'Salary']
#X.shape, Y.shape
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4, random_state=1)
model = linear_model.LinearRegression()
model.fit(X_train.values.reshape(-1, 1), Y_train)
#Y_pred = model.predict(X_test)
print("coffecient", model.coef_)
print("Intercept", model.intercept_)
print("###########################################")
print("Predict Value Of Salary From Rating")
print(model.predict(X_train.values.reshape(-1, 1)))
print("###########################################")
print("IQR Value in the Salary")
print(iqr(df['Salary']))
print("###########################################")
print("IQR Value in the Age")
print(iqr(df['Age']))
print("###########################################")
print("IQR Value in the Rating")
print(iqr(df['Rating']))
print("###########################################")
print("###########################################")
print("###########################################")
print("------------RESEARCH QUITSION & PREDICTION-------------")
print("#################################################################################")
print("What will be the employee's salary next year if he receives a higher Rating?")
print("#################################################################################")
id = input("Enter ID Of The Employee :")
rate = input("Give Rating From (1:5) : ")
while True:
    if int(rate) > 5:
        rate = input("Please Enter Rating From (1:5) : ")
    elif int(rate) <= 5:
        break

for x in range(1, 16):
    if id == df.loc[str(x), 'ID']:
        print('ID :', df.loc[str(x), 'ID'])
        print('Name :', df.loc[str(x), 'Name'])
        print('Salary :', df.loc[str(x), 'Salary'])
        print('Rating :', df.loc[str(x), 'Rating'])
        print('-------------------------------------')
        sum = int(rate)+int(df.loc[str(x), 'Rating'])
        avg = int(sum)/2
        if avg <= int(df.loc[str(x), 'Rating']):
            print('The New Rating Is : ', int(avg))
            print('The Salary Belongs To', df.loc[str(
                x), 'Name'], 'Will Not Be Change Next Year')
            print("###########################################")
            print("###########################################")
            print("###########################################")
            break
        if avg <= 3:
            print('The New Rating Is :', int(avg))
            print('The New Salary Belongs To', df.loc[str(
                x), 'Name'], 'Will Be', df.loc[str(x), 'Salary']+1000, 'Next Year.')
            print("###########################################")
            print("###########################################")
            print("###########################################")

        elif avg <= 4:
            print('The New Rating Is : ', int(avg))
            print('The New Salary Belongs To', df.loc[str(
                x), 'Name'], 'Will Be', df.loc[str(x), 'Salary']+1500, 'Next Year.')
            print("###########################################")
            print("###########################################")
            print("###########################################")
        elif avg == 5:
            print('The New Rating Is : ', int(avg))
            print('The New Salary Belongs To', df.loc[str(
                x), 'Name'], 'Will Be', df.loc[str(x), 'Salary']+2000, 'Next Year.')
            print("###########################################")
            print("###########################################")
            print("###########################################")
        else:
            print('The New Rating Is : ', int(avg))
            print('The Salary Belongs To',
                  df.loc[str(x), 'Name'], 'Will Not Be Change Next Year')
            print("###########################################")
            print("###########################################")
            print("###########################################")
        break
    else:
        print('Waiting....')
        if x == 15:
            print('Data Not Found')
            break

################################################
################################################
################################################
############### 1- BAR CHART####################
plt.bar(df.loc[:, 'ID'], df.loc[:, 'Salary'])
plt.title('Bar Chart Between ID And Salary')
plt.xlabel("ID")
plt.ylabel("Salary")
plt.show()
################################################

################# 2- PIE CHART##################
plt.pie(df.loc[:, 'Salary'], autopct="%.2f", labels=df.loc[:, 'Name'])
plt.title('Pie Chart Of Salary')
plt.show()
###############################################

################ 3- BOX PLOT###################
plt.boxplot(df.loc[:, 'Salary'])
plt.ylabel("Salary")
plt.title('Box Plot Of Salary')
plt.show()
plt.boxplot(df.loc[:, 'Age'])
plt.ylabel("Age")
plt.title('Box Plot Of Age')
plt.show()
plt.boxplot(df.loc[:, 'Rating'])
plt.ylabel("Rating")
plt.title('Box Plot Of Rating')
plt.show()
###############################################

################# 4- SCATTER PLOT##############
plt.scatter(df.loc[:, 'ID'], df.loc[:, 'Salary'])
plt.title('Scatter Plot Between ID And Salary')
plt.xlabel("ID")
plt.ylabel("Salary")
plt.show()
plt.scatter(X_test, Y_test, color='Red')
plt.title('Linear Regression')
plt.plot(X_test, model.predict(X_test.values.reshape(-1, 1)),
         color='Blue', linewidth=3)
plt.show()
#######################################################
######################FINISHED#########################
#######################################################
