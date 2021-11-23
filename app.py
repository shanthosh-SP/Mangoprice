from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
import numpy as np

# import pandas as pd
# import datetime

app = Flask(__name__,template_folder='template')
Load_model = load_model("Mango_Model.h5")


Scaler = pickle.load(open('Mango.pkl', 'rb'))


@app.route("/", methods=["GET", "POST"])
def homey():
    return render_template('entry3.html')
@app.route("/index1", methods=["GET","POST"])
def index():
    return render_template('nifty.html')

@app.route('/method', methods=['POST'])
def predict():
    Value = request.form["Close Price"]
    scaled = Scaler.transform(np.array(Value).reshape(-1, 1))
    prediction = Load_model.predict(scaled.reshape(scaled.shape[0], 1, scaled.shape[1]))
    Result = Scaler.inverse_transform(prediction)[0][0]

    return render_template("nifty.html",prediction_text='The Predicted Mango price is {}'.format(Result))


if __name__ == "__main__":
    app.run(debug=True)