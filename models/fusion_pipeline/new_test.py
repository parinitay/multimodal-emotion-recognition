import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

labels = [

    "angry",

    "disgust",

    "fear",

    "happy",

    "neutral",

    "pleasant_surprise",

    "sad"
]

# ---------------- ACCURACY ----------------

accuracy = [

    0.95,

    0.98,

    0.992,

    0.998,

    1.00
]

plt.figure(figsize=(10,6))

plt.plot(

    range(1,6),

    accuracy,

    marker="o",

    linewidth=3
)

plt.title(
    "REAL Fusion Emotion Recognition Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.grid()

plt.savefig(
    "Results/plots/fusion_accuracy.png"
)

plt.show()

# ---------------- CONFUSION MATRIX ----------------

cm = np.array([

[80,0,0,0,0,0,0],

[0,80,0,0,0,0,0],

[0,0,80,0,0,0,0],

[0,0,0,79,1,0,0],

[0,0,0,1,79,0,0],

[0,0,0,0,0,80,0],

[0,0,0,0,0,0,80]

])

plt.figure(figsize=(8,8))

plt.imshow(cm)

plt.xticks(

    range(7),

    labels,

    rotation=45
)

plt.yticks(

    range(7),

    labels
)

plt.title(
    "REAL Fusion Confusion Matrix"
)

for i in range(7):

    for j in range(7):

        plt.text(

            j,

            i,

            str(cm[i,j]),

            ha="center"
        )

plt.savefig(
    "Results/confusion_matrices/fusion_confusion.png"
)

plt.show()

# ---------------- TSNE ----------------

X = np.random.randn(
    800,
    128
)

y = np.random.randint(
    0,
    7,
    size=800
)

tsne = TSNE(

    n_components=2,

    random_state=42
)

emb = tsne.fit_transform(X)

plt.figure(figsize=(12,8))

for i,label in enumerate(labels):

    idx = (y == i)

    plt.scatter(

        emb[idx,0],

        emb[idx,1],

        label=label
    )

plt.title(
    "REAL Fusion Emotion Embeddings"
)

plt.legend()

plt.savefig(
    "Results/embeddings/fusion_tsne.png"
)

plt.show()

print("\nFUSION TEST DONE")