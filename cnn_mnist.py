"""
CodeAlpha ML Internship - Task 3: Handwritten Character Recognition (CNN)
Dataset: MNIST (digits) by default, or EMNIST (letters) via --dataset emnist
Run   : python cnn_mnist.py                  # MNIST, ~99% accuracy
        python cnn_mnist.py --dataset emnist # needs: pip install tensorflow-datasets
Needs : pip install tensorflow matplotlib scikit-learn seaborn
(requires internet on first run to download the dataset)
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix, classification_report

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
ap.add_argument("--epochs", type=int, default=15)
args = ap.parse_args()
tf.keras.utils.set_random_seed(42)


def load_emnist_letters():
    import tensorflow_datasets as tfds
    tr, te = tfds.as_numpy(tfds.load("emnist/letters", split=["train", "test"], batch_size=-1))
    # EMNIST images are stored transposed; fix orientation. Labels are 1..26 -> 0..25
    f = lambda d: (np.transpose(d["image"], (0, 2, 1, 3)), d["label"] - 1)
    (xtr, ytr), (xte, yte) = f(tr), f(te)
    names = [chr(65 + i) for i in range(26)]
    return xtr, ytr, xte, yte, names


if args.dataset == "mnist":
    (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    xtr, xte = xtr[..., None], xte[..., None]
    names = [str(i) for i in range(10)]
else:
    xtr, ytr, xte, yte, names = load_emnist_letters()

xtr, xte = xtr.astype("float32") / 255.0, xte.astype("float32") / 255.0
n_classes = len(names)
print("Train:", xtr.shape, "Test:", xte.shape, "Classes:", n_classes)

model = models.Sequential([
    layers.Input((28, 28, 1)),
    layers.RandomRotation(0.08), layers.RandomZoom(0.1), layers.RandomTranslation(0.08, 0.08),  # augmentation
    layers.Conv2D(32, 3, padding="same", activation="relu"), layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D(), layers.Dropout(0.25),
    layers.Conv2D(64, 3, padding="same", activation="relu"), layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(), layers.Dropout(0.25),
    layers.Flatten(), layers.Dense(128, activation="relu"), layers.Dropout(0.4),
    layers.Dense(n_classes, activation="softmax"),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

hist = model.fit(xtr, ytr, validation_split=0.1, epochs=args.epochs, batch_size=128,
                 callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                            callbacks.ReduceLROnPlateau(patience=2, factor=0.5)])

loss, acc = model.evaluate(xte, yte, verbose=0)
print(f"\nTest accuracy: {acc:.4f}")
pred = model.predict(xte, verbose=0).argmax(1)
print(classification_report(yte, pred, target_names=names, digits=4))

# plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(hist.history["accuracy"], label="train"); ax[0].plot(hist.history["val_accuracy"], label="val")
ax[0].set_title("Accuracy"); ax[0].legend()
ax[1].plot(hist.history["loss"], label="train"); ax[1].plot(hist.history["val_loss"], label="val")
ax[1].set_title("Loss"); ax[1].legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=120); plt.close()

plt.figure(figsize=(8, 7))
sns.heatmap(confusion_matrix(yte, pred), cmap="Blues", xticklabels=names, yticklabels=names,
            annot=n_classes <= 10, fmt="d")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

wrong = np.where(pred != yte)[0][:12]
fig, axs = plt.subplots(2, 6, figsize=(12, 4.5))
for a, i in zip(axs.ravel(), wrong):
    a.imshow(xte[i].squeeze(), cmap="gray"); a.axis("off")
    a.set_title(f"true {names[yte[i]]} / pred {names[pred[i]]}", fontsize=9)
plt.tight_layout(); plt.savefig("misclassified.png", dpi=120); plt.close()

model.save(f"handwriting_cnn_{args.dataset}.keras")
print("Saved model + plots")
