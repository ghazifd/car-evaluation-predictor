from flask import Flask, request, render_template
import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd


app = Flask(__name__)
MODEL_PATH = "model/model.joblib"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Modèle manquant. Lance d’abord: python train_model.py")

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    # On recrée un DataFrame avec les bons noms de colonnes
    input_df = pd.DataFrame([{
        "buying": data.get("buying"),
        "maint": data.get("maint"),
        "doors": data.get("doors"),
        "persons": data.get("persons"),
        "lug_boot": data.get("lug_boot"),
        "safety": data.get("safety"),
    }])

    # Prédiction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    labels = model.classes_
    probs = dict(zip(labels, proba))

    # --- Générer un graphique ---
    plt.figure(figsize=(6,4))
    plt.bar(probs.keys(), probs.values(), color="skyblue")
    plt.title("Probabilités de classification")
    plt.ylabel("Score")
    plt.xlabel("Classe")
    plt.ylim(0, 1)

    if not os.path.exists("static"):
        os.makedirs("static")

    chart_path = "static/probabilities.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template(
        "result.html",
        prediction=prediction,
        probs=probs,
        chart_path=chart_path
    )


if __name__ == "__main__":
    app.run(debug=True)
