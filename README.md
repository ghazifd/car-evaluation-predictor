
# 🚗 Car Evaluation Prediction Web App

Ce projet est une application complète de machine learning avec une interface web permettant de prédire la classe d’acceptabilité d’un véhicule à partir de ses caractéristiques. Il a été développé dans le cadre d’un stage d’application.

---

## 🎯 Objectif

L’objectif de ce projet est de permettre à un utilisateur d’entrer les caractéristiques d’un véhicule (prix, sécurité, capacité, etc.) et d’obtenir automatiquement une prédiction sur sa qualité globale : **unacceptable**, **acceptable**, **good** ou **very good**.

---

## 📊 Données utilisées

Le dataset utilisé est **Car Evaluation Data Set**, disponible sur l'UCI Machine Learning Repository. Il comporte 1728 instances avec les caractéristiques suivantes :

- `buying`: prix d’achat (low, med, high, vhigh)
- `maint`: coût d’entretien (low, med, high, vhigh)
- `doors`: nombre de portes (2, 3, 4, 5more)
- `persons`: nombre de personnes (2, 4, more)
- `lug_boot`: taille du coffre (small, med, big)
- `safety`: niveau de sécurité (low, med, high)
- `class`: variable cible (unacc, acc, good, vgood)

---

## 🧠 Modèle de Machine Learning

- Modèle utilisé : `RandomForestClassifier` (100 arbres)
- Encodage des variables catégorielles via un `OrdinalEncoder` manuel
- Données séparées en **67% train / 33% test**
- Évaluation via `accuracy_score`, `classification_report`, `confusion_matrix`

---

## 🌐 Interface Web (Front-end)

Développée avec **Flask**, l’interface permet à l’utilisateur de :

1. Sélectionner les caractéristiques du véhicule via un formulaire web
2. Envoyer les données au modèle
3. Obtenir une prédiction de la classe affichée à l’écran

---

## 🗂 Arborescence du projet

```
car-evaluation-predictor/
├── model/
│   └── model.pkl
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── Prediction.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ Lancer le projet en local

```bash
git clone https://github.com/<ton-utilisateur>/car-evaluation-predictor.git
cd car-evaluation-predictor
pip install -r requirements.txt
python app.py
```

---

## 🧪 Exemple d’utilisation

```text
Buying      : high
Maint       : med
Doors       : 4
Persons     : 4
Lug_boot    : big
Safety      : high

Résultat → Classe prédite : vgood
```

---

## 📦 Dépendances

- flask
- numpy
- pandas
- scikit-learn

---

## 👨‍💻 Réalisé par

**Ghazi FOUDHAILI**  
Stage chez **YottaByte**  
Période : Juillet - Août 2025

