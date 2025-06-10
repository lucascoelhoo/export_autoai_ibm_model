from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carrega o pipeline
pipeline = joblib.load("modelo-exportado-pipeline-autoai.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Espera um JSON com "inputs": [[...], [...]]
    inputs = np.array(data["inputs"])
    preds = pipeline.predict(inputs)
    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# curl -X POST http://localhost:5000/predict \
#      -H "Content-Type: application/json" \
#      -d '{"inputs": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]}'

    # pip3 install autoai-libs==2.0.* xgboost==2.0.* scikit-learn==1.3.* numpy


