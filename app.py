from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# loading model + scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("heartdisease.html")

@app.route("/predict", methods=["POST"])
def predict():

    # getting input data
    data = request.get_json()

    # convert to dataframe
    df = pd.DataFrame([data])

    # match training columns
    df = df.reindex(columns=columns, fill_value=0)

    # scaling
    df_scaled = scaler.transform(df)

    # prediction
    prediction = model.predict(df_scaled)[0]

    # convert into human-readable message
    if prediction == 1:
        result_text = "⚠️ High risk of heart disease"
    else:
        result_text = "✅ Low risk of heart disease"

    return jsonify({
        "prediction": int(prediction),
        "result_text": result_text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)