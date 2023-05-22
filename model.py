import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json

app = Flask(__name__)
# dataset = pd.read_csv('Salary_Data.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 1].values
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
#
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# pickle.dump(regressor, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
   value  = int(request.headers['value'])
   pred= model.predict([[value]])
   return str(pred)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
