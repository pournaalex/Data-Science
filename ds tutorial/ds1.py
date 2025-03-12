import pandas as pd  #importing pandas
import statsmodels.api as sm  #importing statsmodel
df = pd.read_csv("setadv.csv")  #loading dataset
X = df[['TV', 'radio', 'newspaper']]  #setting independant variables
y = df['sales']  #setting dependant variables
X = sm.add_constant(X)  #adding intercept
model = sm.OLS(y, X).fit()  #fitting
rse = model.mse_resid ** 0.5  #rse
r_squared = model.rsquared    #r2
f_statistic = model.fvalue    #fstat
print("Residual Standard Error(RSE): "+str(round(rse, 4)))  #output1
print("R-squared(R²): "+str(round(r_squared, 4)))  #output2
print("F-statistic: "+str(round(f_statistic, 4)))  #output3

