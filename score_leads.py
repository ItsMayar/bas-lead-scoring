"""Lead scoring: trains a logistic regression on lead history and ranks leads by conversion likelihood."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

df = pd.read_csv("sample_leads.csv")
X, y = df.drop(columns="converted"), df["converted"]
cat = ["service", "company_size", "source"]
num = ["contacts_made", "days_since_contact", "meeting_held"]
pre = ColumnTransformer([("c", OneHotEncoder(), cat), ("n", StandardScaler(), num)])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]
print(f"Accuracy: {accuracy_score(yte, p > .5):.2f}  ROC-AUC: {roc_auc_score(yte, p):.2f}")
print(f"Baseline (always predict majority): {max(yte.mean(), 1-yte.mean()):.2f}")

# Score all leads and rank
df["score"] = (model.predict_proba(X)[:, 1] * 100).round(0)
df.sort_values("score", ascending=False).to_csv("ranked_leads.csv", index=False)

# Which factors matter
names = model.named_steps["pre"].get_feature_names_out()
coef = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()
coef.plot.barh(figsize=(8, 6), title="Factors affecting conversion likelihood (sample data)")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=120)
print(coef.tail(3))
