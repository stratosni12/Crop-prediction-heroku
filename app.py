import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def submit():
    #HMTL -> .py
    if request.method == "POST":
        humid = request.form["humidity"]
        rain = request.form["rainfall"]
        user_input = [humid, rain] 
        #print(user_input)
        result = model.predict([user_input])[0]
        print(result)
    
    #.py -> HTML
    return render_template("index.html", prediction_text='The most suitable crop is: {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)

