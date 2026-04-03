"""
Full Train & Test Pipeline — Chemical Environment Dataset
=========================================================
Targets:
  1. is_env_relevant         -> Binary Classification
  2. pollutant_concentration -> Regression (log-scaled)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib, os

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# ── Paths (FIXED FOR YOUR WINDOWS MACHINE) ───────────────────────────────────
CSV = r"C:\Users\hp\OneDrive\Videos\Desktop\blender-project\full_ml_dataset_scaled.csv"
OUT = r"C:\Users\hp\OneDrive\Videos\Desktop\blender-project"
os.makedirs(OUT, exist_ok=True)

print("=" * 65)
print("  CHEMICAL ENVIRONMENT ML — TRAIN & TEST PIPELINE")
print("=" * 65)
print(f"\n[INFO] Loading data from:\n       {CSV}\n")

# ── 1. Load & Prepare ────────────────────────────────────────────────────────
df = pd.read_csv(CSV)

DROP = [
    "chemical_compound", "chemical_structure", "classification",
    "physical_properties", "linked_sg_pollutant",
    "classification_encoded.1", "classification_encoded_scaled.1"
]
df = df.drop(columns=[c for c in DROP if c in df.columns])

TARGET_C = "is_env_relevant"
TARGET_R = "pollutant_concentration"
EXCLUDE  = {TARGET_C, TARGET_R, "classification_encoded", "year"}
FEATS    = [c for c in df.columns if c not in EXCLUDE]

X   = df[FEATS].fillna(0)
y_c = df[TARGET_C].astype(int)
y_r = np.log1p(df[TARGET_R].astype(float))

print(f"[INFO] Dataset shape   : {df.shape}")
print(f"[INFO] Features used   : {X.shape[1]}")
print(f"[INFO] Total samples   : {X.shape[0]}")
print(f"[INFO] Class balance   : 0={(y_c==0).sum()}  1={(y_c==1).sum()}")

# ── 2. Train / Test Split (80/20) ────────────────────────────────────────────
X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
    X, y_c, y_r, test_size=0.2, random_state=42, stratify=y_c)

print(f"\n[INFO] Train size: {len(X_tr)}  |  Test size: {len(X_te)}")

# =============================================================================
#  CLASSIFICATION
# =============================================================================
clf_models = {
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "SVM":                 SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
}

print("\n" + "=" * 65)
print("  TRAINING — Classification  (is_env_relevant)")
print("=" * 65)

clf_res = {}
for name, model in clf_models.items():

    model.fit(X_tr, yc_tr)
    train_preds  = model.predict(X_tr)
    train_probas = model.predict_proba(X_tr)[:, 1]
    train_acc    = accuracy_score(yc_tr, train_preds)
    train_auc    = roc_auc_score(yc_tr, train_probas)

    test_preds  = model.predict(X_te)
    test_probas = model.predict_proba(X_te)[:, 1]
    test_acc    = accuracy_score(yc_te, test_preds)
    test_auc    = roc_auc_score(yc_te, test_probas)
    test_prec   = precision_score(yc_te, test_preds)
    test_rec    = recall_score(yc_te, test_preds)
    test_f1     = f1_score(yc_te, test_preds)

    cv = cross_validate(
        model, X, y_c,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring=["accuracy", "roc_auc"], n_jobs=-1
    )

    clf_res[name] = dict(
        model=model,
        train_acc=train_acc, train_auc=train_auc,
        test_acc=test_acc,   test_auc=test_auc,
        test_prec=test_prec, test_rec=test_rec, test_f1=test_f1,
        test_preds=test_preds, test_probas=test_probas,
        cv_acc=cv["test_accuracy"].mean(),
        cv_auc=cv["test_roc_auc"].mean()
    )

    print(f"\n  [{name}]")
    print(f"    TRAIN  ->  Acc={train_acc:.4f}   AUC={train_auc:.4f}")
    print(f"    TEST   ->  Acc={test_acc:.4f}   AUC={test_auc:.4f}   Prec={test_prec:.4f}   Rec={test_rec:.4f}   F1={test_f1:.4f}")
    print(f"    CV(5)  ->  Acc={cv['test_accuracy'].mean():.4f}   AUC={cv['test_roc_auc'].mean():.4f}")

best_c = max(clf_res, key=lambda k: clf_res[k]["test_auc"])
print(f"\n  BEST Classifier -> {best_c}  (Test AUC={clf_res[best_c]['test_auc']:.4f})")

# =============================================================================
#  REGRESSION
# =============================================================================
reg_models = {
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    "Ridge":             Ridge(alpha=1.0),
    "SVR":               SVR(kernel="rbf", C=10, epsilon=0.1),
}

print("\n" + "=" * 65)
print("  TRAINING — Regression  (pollutant_concentration)")
print("=" * 65)

reg_res = {}
for name, model in reg_models.items():

    model.fit(X_tr, yr_tr)
    train_preds = model.predict(X_tr)
    train_r2    = r2_score(yr_tr, train_preds)
    train_rmse  = np.sqrt(mean_squared_error(yr_tr, train_preds))

    test_preds = model.predict(X_te)
    test_r2    = r2_score(yr_te, test_preds)
    test_rmse  = np.sqrt(mean_squared_error(yr_te, test_preds))
    test_mae   = mean_absolute_error(yr_te, test_preds)

    cv = cross_validate(
        model, X, y_r,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring=["r2", "neg_root_mean_squared_error"], n_jobs=-1
    )

    reg_res[name] = dict(
        model=model,
        train_r2=train_r2, train_rmse=train_rmse,
        test_r2=test_r2,   test_rmse=test_rmse, test_mae=test_mae,
        test_preds=test_preds,
        cv_r2=cv["test_r2"].mean(),
        cv_rmse=-cv["test_neg_root_mean_squared_error"].mean()
    )

    print(f"\n  [{name}]")
    print(f"    TRAIN  ->  R2={train_r2:.4f}   RMSE={train_rmse:.4f}")
    print(f"    TEST   ->  R2={test_r2:.4f}   RMSE={test_rmse:.4f}   MAE={test_mae:.4f}")
    print(f"    CV(5)  ->  R2={cv['test_r2'].mean():.4f}   RMSE={-cv['test_neg_root_mean_squared_error'].mean():.4f}")

best_r = max(reg_res, key=lambda k: reg_res[k]["test_r2"])
print(f"\n  BEST Regressor -> {best_r}  (Test R2={reg_res[best_r]['test_r2']:.4f})")

# ── Save Models ───────────────────────────────────────────────────────────────
clf_path = os.path.join(OUT, "best_classifier.pkl")
reg_path = os.path.join(OUT, "best_regressor.pkl")
joblib.dump(clf_res[best_c]["model"], clf_path)
joblib.dump(reg_res[best_r]["model"], reg_path)
print(f"\n[SAVED] Classifier -> {clf_path}")
print(f"[SAVED] Regressor  -> {reg_path}")

# =============================================================================
#  VISUALISATION
# =============================================================================
print("\n[INFO] Generating report charts...")

PAL = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
BG  = "#0F172A"
BG2 = "#1E293B"
TXT = "white"
SUB = "#94A3B8"

fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)
TK  = dict(color=TXT, fontsize=11, fontweight="bold", pad=8)

def style(ax):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=SUB, labelsize=8)
    ax.spines[:].set_visible(False)

names  = list(clf_res.keys())
rnames = list(reg_res.keys())
x  = np.arange(len(names))
rx = np.arange(len(rnames))
w  = 0.35

ax = fig.add_subplot(gs[0, 0])
style(ax)
ax.bar(x - w/2, [clf_res[n]["train_acc"] for n in names], w, label="Train", color="#3B82F6", alpha=0.85)
ax.bar(x + w/2, [clf_res[n]["test_acc"]  for n in names], w, label="Test",  color="#10B981", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7, color=SUB)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Accuracy", color=SUB)
ax.set_title("Train vs Test Accuracy", **TK)
ax.legend(fontsize=8, labelcolor=TXT, facecolor=BG, edgecolor="none")

ax = fig.add_subplot(gs[0, 1])
style(ax)
ax.bar(x - w/2, [clf_res[n]["train_auc"] for n in names], w, label="Train", color="#3B82F6", alpha=0.85)
ax.bar(x + w/2, [clf_res[n]["test_auc"]  for n in names], w, label="Test",  color="#10B981", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7, color=SUB)
ax.set_ylim(0, 1.15)
ax.set_ylabel("AUC", color=SUB)
ax.set_title("Train vs Test AUC", **TK)
ax.legend(fontsize=8, labelcolor=TXT, facecolor=BG, edgecolor="none")

ax = fig.add_subplot(gs[0, 2])
style(ax)
cv_aucs = [clf_res[n]["cv_auc"] for n in names]
bars = ax.bar(names, cv_aucs, color=PAL)
ax.set_ylim(0, 1.15)
ax.set_title("Cross-Validation AUC (5-Fold)", **TK)
ax.set_ylabel("AUC", color=SUB)
ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7, color=SUB)
for bar, v in zip(bars, cv_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
            ha="center", color=TXT, fontsize=9, fontweight="bold")

ax = fig.add_subplot(gs[1, 0])
style(ax)
for (name, res), col in zip(clf_res.items(), PAL):
    fpr, tpr, _ = roc_curve(yc_te, res["test_probas"])
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} ({res['test_auc']:.3f})")
ax.plot([0, 1], [0, 1], "w--", lw=0.8, alpha=0.4)
ax.set_title("ROC Curves (Test Set)", **TK)
ax.set_xlabel("False Positive Rate", color=SUB)
ax.set_ylabel("True Positive Rate", color=SUB)
ax.legend(fontsize=7, labelcolor=TXT, facecolor=BG, edgecolor="none")

ax = fig.add_subplot(gs[1, 1])
style(ax)
cm = confusion_matrix(yc_te, clf_res[best_c]["test_preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=0.5, linecolor=BG,
            annot_kws={"size": 14, "color": TXT})
ax.set_title(f"Confusion Matrix — {best_c}", **TK)
ax.set_xlabel("Predicted", color=SUB)
ax.set_ylabel("Actual", color=SUB)
ax.set_xticklabels(["Not Env", "Env Relevant"], color=SUB)
ax.set_yticklabels(["Not Env", "Env Relevant"], color=SUB, rotation=0)

ax = fig.add_subplot(gs[1, 2])
style(ax)
ax.axis("off")
metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
train_v = [f"{clf_res[best_c]['train_acc']:.4f}", f"{clf_res[best_c]['train_auc']:.4f}", "-", "-", "-"]
test_v  = [f"{clf_res[best_c]['test_acc']:.4f}", f"{clf_res[best_c]['test_auc']:.4f}",
           f"{clf_res[best_c]['test_prec']:.4f}", f"{clf_res[best_c]['test_rec']:.4f}",
           f"{clf_res[best_c]['test_f1']:.4f}"]
table = ax.table(
    cellText=[[m, t, v] for m, t, v in zip(metrics, train_v, test_v)],
    colLabels=["Metric", "Train", "Test"],
    cellLoc="center", loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
for (r, c), cell in table.get_celld().items():
    cell.set_facecolor("#3B82F6" if r == 0 else BG2)
    cell.set_text_props(color=TXT)
    cell.set_edgecolor(BG)
ax.set_title(f"Classification Metrics — {best_c}", **TK)

ax = fig.add_subplot(gs[2, 0])
style(ax)
ax.bar(rx - w/2, [reg_res[n]["train_r2"] for n in rnames], w, label="Train", color="#3B82F6", alpha=0.85)
ax.bar(rx + w/2, [reg_res[n]["test_r2"]  for n in rnames], w, label="Test",  color="#10B981", alpha=0.85)
ax.set_xticks(rx)
ax.set_xticklabels([n.replace(" ", "\n") for n in rnames], fontsize=7, color=SUB)
ax.set_ylabel("R2", color=SUB)
ax.set_title("Train vs Test R2  (Regression)", **TK)
ax.legend(fontsize=8, labelcolor=TXT, facecolor=BG, edgecolor="none")
ax.axhline(0, color="white", lw=0.8, linestyle="--", alpha=0.5)

ax = fig.add_subplot(gs[2, 1])
style(ax)
bp = reg_res[best_r]["test_preds"]
ax.scatter(yr_te, bp, alpha=0.55, s=25, color="#3B82F6", edgecolors="none")
mn, mx = min(yr_te.min(), bp.min()), max(yr_te.max(), bp.max())
ax.plot([mn, mx], [mn, mx], "w--", lw=1.2)
ax.set_title(f"Actual vs Predicted — {best_r}", **TK)
ax.set_xlabel("Actual (log scale)", color=SUB)
ax.set_ylabel("Predicted (log scale)", color=SUB)
ax.text(0.05, 0.9, f"R2={reg_res[best_r]['test_r2']:.4f}",
        transform=ax.transAxes, color=TXT, fontsize=11)

ax = fig.add_subplot(gs[2, 2])
style(ax)
residuals = yr_te.values - bp
ax.scatter(bp, residuals, alpha=0.55, s=25, color="#F59E0B", edgecolors="none")
ax.axhline(0, color="white", lw=1.2, linestyle="--")
ax.set_title("Residual Plot", **TK)
ax.set_xlabel("Predicted", color=SUB)
ax.set_ylabel("Residual", color=SUB)

ax = fig.add_subplot(gs[3, :2])
style(ax)
if hasattr(clf_res[best_c]["model"], "feature_importances_"):
    fi = pd.Series(clf_res[best_c]["model"].feature_importances_, index=FEATS).nlargest(15).sort_values()
    ax.barh(fi.index, fi.values, color="#3B82F6", edgecolor="none")
    ax.set_title(f"Top-15 Feature Importances — {best_c}", **TK)
    ax.set_xlabel("Importance", color=SUB)

ax = fig.add_subplot(gs[3, 2])
style(ax)
ax.axis("off")
rows = [[n,
         f"{reg_res[n]['train_r2']:.3f}",
         f"{reg_res[n]['test_r2']:.3f}",
         f"{reg_res[n]['test_rmse']:.3f}",
         f"{reg_res[n]['test_mae']:.3f}"] for n in rnames]
tbl = ax.table(cellText=rows,
               colLabels=["Model", "Train R2", "Test R2", "RMSE", "MAE"],
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#10B981" if r == 0 else BG2)
    cell.set_text_props(color=TXT)
    cell.set_edgecolor(BG)
ax.set_title("Regression Summary", **TK)

fig.suptitle("Chemical Environment — Full Train & Test Report",
             color=TXT, fontsize=17, fontweight="bold", y=1.01)

out_png = os.path.join(OUT, "train_test_report.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"[SAVED] Report -> {out_png}")

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  DETAILED CLASSIFICATION REPORT (Test Set)")
print("=" * 65)
print(classification_report(yc_te, clf_res[best_c]["test_preds"],
      target_names=["Not Env Relevant", "Env Relevant"]))

print("=" * 65)
print("  OVERFITTING CHECK")
print("=" * 65)
for name in clf_res:
    gap    = clf_res[name]["train_acc"] - clf_res[name]["test_acc"]
    status = "OK - No overfit" if abs(gap) < 0.05 else "WARNING - Overfit" if gap > 0.05 else "WARNING - Underfit"
    print(f"  {name:22s}  Gap={gap:+.4f}  ->  {status}")

print("\n" + "=" * 65)
print("  ALL DONE! Files saved to:")
print(f"  {OUT}")
print("=" * 65)
