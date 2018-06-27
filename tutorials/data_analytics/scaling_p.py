import pandas as pd

#import the csv files.
x_train = pd.read_csv(
    r'D:\ddos_attacks.csv')
#y_train = pd.read_csv(
#    r'C:\Users\Sai Kamat\source\repos\Python_for_DS\Datasets\loan_prediction\Y_train.csv')
#x_test = pd.read_csv(
    #r'C:\Users\Sai Kamat\source\repos\Python_for_DS\Datasets\loan_prediction\X_test.csv')
#y_test = pd.read_csv(
   # r'C:\Users\Sai Kamat\source\repos\Python_for_DS\Datasets\loan_prediction\Y_test.csv')

#print(x_train.head())

#feature scaling
import matplotlib.pyplot as plt


x_train[x_train.dtypes[(x_train.dtypes == 'float64')|(x_train.dtypes == 'int64')].index.values].hist(figsize=[11,11])
plt.show()
