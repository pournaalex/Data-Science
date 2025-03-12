import numpy as np #importing numpy
from sklearn.model_selection import train_test_split #importing trainttestsplit
from sklearn.linear_model import LogisticRegression #importing logistic regression
from sklearn.metrics import accuracy_score #importing accuracy score
data={
    'days attended': [100, 98, 90, 80, 80, 60, 83, 73, 92, 68, 65],
    'work completion': [40, 85, 90, 60, 75, 50, 80, 65, 88, 72, 62],
    'salary increase (Y)': [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
} #dataset
X= np.array(list(zip(data['days attended'], data['work completion']))) #data to array
y= np.array(data['salary increase (Y)'])
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42) #splitting into train data and test data
model= LogisticRegression() #calling regression model
model.fit(X_train, y_train) #fitting the model
y_pred= model.predict(X_test) #test predictions
accuracy= accuracy_score(y_test, y_pred) #evaluating accuracy
print(f"Model Accuracy: {accuracy}") #printing accuracy
unknown= np.array([[79, 62], [83, 68]]) #unknown input
predictions= model.predict(unknown) #predicting for unknown input
print("Predictions for unknown employees:") #output
i=1 #initializing i 
for pred in predictions: #for loop starts
    print("Employee", i, ":", pred) #output
    i+=1 #incrementing i

