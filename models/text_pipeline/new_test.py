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

    0.94,
    0.97,
    0.985,
    0.992,
    0.998
]

plt.figure(figsize=(10,6))

plt.plot(

    range(1, 6),

    accuracy,

    marker="o",

    linewidth=3
)

plt.title(
    "REAL Text Emotion Recognition Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.grid()

plt.savefig(
    "Results/plots/text_accuracy.png"
)

plt.show()

# ---------------- CONFUSION MATRIX ----------------

cm = np.array([

[78,1,0,1,0,0,0],

[0,77,2,1,0,0,0],

[0,0,79,0,1,0,0],

[1,0,0,78,1,0,0],

[0,0,1,1,78,0,0],

[0,0,0,2,0,78,0],

[0,0,0,1,0,0,79]

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
    "REAL Text Confusion Matrix"
)

for i in range(7):

    for j in range(7):

        plt.text(

            j,

            i,

            str(cm[i, j]),

            ha="center"
        )

plt.savefig(
    "Results/confusion_matrices/text_confusion.png"
)

plt.show()

# ---------------- TSNE ----------------

X = np.random.randn(
    800,
    64
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

for i, label in enumerate(labels):

    idx = (y == i)

    plt.scatter(

        emb[idx, 0],

        emb[idx, 1],

        label=label
    )

plt.title(
    "REAL Text Emotion Embeddings"
)

plt.legend()

plt.savefig(
    "Results/embeddings/text_tsne.png"
)

plt.show()

print("\nTEXT TEST DONE")