from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pandas as pd

"""
Create a simple rest api with flask.
The user gives an json with the values: ex

dic {
    "X": 1,
    "Y": 2,
    "month": 3,
    "day": 10,
    "FFMC": 4,
    "DMC": 5,
    "DC": 7,
    "ISI": 8,
    "temp": 9,
    "RH": 10,
    "wind": 11,
    "rain": 12  
    }
"""
app = Flask(__name__)

CORS(app, origins=["http://localhost:5299/"])

#load the model
NN_model = tf.keras.models.load_model("NN_ForestFires.h5")


@app.route("/make-prediction-NN", methods=["POST"])
def makePrediction():
    #get the data as an json
    data = request.get_json()

    data = {
        "X": data["X"],
        "Y": data["Y"],
        "month": data["month"],
        "day": data["day"],
        "FFMC": data["FFMC"],
        "DMC": data["DMC"],
        "DC": data["DC"],
        "ISI": data["ISI"],
        "temp": data["temp"],
        "RH": data["RH"],
        "wind": data["wind"],
        "rain": data["rain"],
    }
    #create df to use in neural netwrok for prediction
    df = pd.DataFrame([data])
    predicted_value = NN_model.predict(df)
    predicted = predicted_value.tolist()
    return jsonify({"predicted": predicted}), 201

if __name__ == '__main__':
    app.run(debug=True)
