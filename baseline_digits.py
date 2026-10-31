"""
Offline baseline for Task 3 (no TensorFlow / no internet needed).
Uses scikit-learn's bundled 8x8 handwritten digits (a small UCI digits set) to verify the
pipeline: image preprocessing -> model -> evaluation. The real submission model is cnn_mnist.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

d = load_digits()
X, y = d.data / 16.0, d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
    "MLP": make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=500, random_state=42)),
}
best, best_acc = None, 0
for n, m in models.items():
    m.fit(Xtr, ytr)
    acc = accuracy_score(yte, m.predict(Xte))
    print(f"{n:10s} test accuracy: {acc:.4f}")
    if acc > best_acc: best, best_acc, best_name = m, acc, n

pred = best.predict(Xte)
print(f"\nBest: {best_name}\n", classification_report(yte, pred, digits=4))
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(yte, pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Digits baseline: {best_name}")
plt.tight_layout(); plt.savefig("baseline_confusion_matrix.png", dpi=120)
