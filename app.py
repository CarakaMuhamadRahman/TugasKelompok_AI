from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
@cross_origin
def home():
    return  'This API for Playing Golf Reccomendation Application'


@cross_origin 
@app.route("/predict", methods=["POST"])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({'Prediction': list(prediction)})

if __name__ == "__main__":
    app.run(debug=False)