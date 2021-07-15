import numpy as np # linear algebra
import pandas as pd
import pickle

df=pd.read_csv('F:\VSCode\Flask_Heroku_deploy\Crop_recommendation.csv') #read input file
df.head()

y = df.label #select label as target for prediction
# only keep humidity and rainfall column to train
X = df.drop(['N', 'P', 'K', 'temperature','ph','label'], axis = 1)
X.head()

#Split data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)

#import randomforest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=4,n_estimators=100,random_state=42).fit(X_train, y_train)
print('RF Accuracy on training set: {:.2f}'.format(model.score(X_train, y_train)))
print('RF Accuracy on test set: {:.2f}'.format(model.score(X_test, y_test)))

#User input
print("Enter your own data to test the model:")
humidity = int(input("Enter Humidity:")) 
rainfall = int(input("Enter Rainfall(mm):")) 
user_input = [humidity, rainfall] 
print(user_input)

result = model.predict([user_input])[0]
print(result)

# Saving model to disk
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))

#Load model
loaded_model = pickle.load(open(filename, 'rb'))
result1 = loaded_model.score(X_test, y_test)
print(result1)
result2 = model.predict([user_input])[0]
print(result2)
