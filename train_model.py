from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib, json

DATA_PATH = Path("data/car.data")
COLUMNS = ["buying","maint","doors","persons","lug_boot","safety","class"]

if not DATA_PATH.exists():
    raise SystemExit("Dataset manquant: data/car.data")

df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
X = df.drop(columns=["class"])
y = df["class"]

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
prep = ColumnTransformer([("cat", enc, list(X.columns))], remainder="drop")

cands = []
cands.append(("RandomForest", Pipeline([("prep", prep), ("clf", RandomForestClassifier(n_estimators=120, random_state=42))])))
cands.append(("SVM-RBF", Pipeline([("prep", prep), ("clf", SVC(kernel="rbf", probability=True, random_state=42, C=2.0, gamma='scale'))])))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

best = None
best_f1 = -1
scores = []

for name, pipe in cands:
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    f1 = f1_score(yte, yp, average="macro")
    acc = accuracy_score(yte, yp)
    scores.append({"model": name, "f1_macro": float(f1), "accuracy": float(acc)})
    if f1 > best_f1:
        best_f1, best = f1, (name, pipe)

best_name, best_pipe = best
Path("model").mkdir(exist_ok=True, parents=True)
Path("metrics").mkdir(exist_ok=True, parents=True)
joblib.dump(best_pipe, Path("model/model.joblib"))
(Path("metrics/metrics.json")).write_text(json.dumps({"scores": scores, "best_model": best_name, "best_f1": best_f1}, indent=2), encoding="utf-8")
print(f"Best model: {best_name} (F1={best_f1:.3f}) -> model/model.joblib")
