from flask import Flask, request, jsonify
import os
from model import predict

app = Flask(__name__)

API_NAME = os.getenv("API_NAME", "ML Prediction API")

@app.route("/")
def home():
    return {"message": API_NAME}

@app.route("/predict", methods=["POST"])
def prediction():
    data = request.get_json()
    value = data["value"]

    result = predict(value)

    return jsonify({
        "input": value,
        "prediction": result
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)