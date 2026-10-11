
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained Netflix model
model = joblib.load("artifacts/netflix_model.joblib")
encoder = joblib.load("artifacts/label_encoder.joblib")


@app.route("/")
def home():
    return jsonify({
        "message": "Netflix Movie and TV Show Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    required = ["rating", "country", "duration", "listed_in"]
    missing = [key for key in required if key not in data]

    if missing:
        return jsonify({
            "error": "Missing fields",
            "fields": missing
        }), 400

    text = (
        f"Rating: {data['rating']} "
        f"Country: {data['country']} "
        f"Duration: {data['duration']} "
        f"Genre: {data['listed_in']}"
    )

    prediction = model.predict([text])
    label = encoder.inverse_transform(prediction)[0]

    return jsonify({"predicted_type": label})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
