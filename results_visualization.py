import numpy as np
import matplotlib.pyplot as plt

train_acc = np.load(
    "Results/tables/speech_train_accuracy.npy"
)

test_acc = np.load(
    "Results/tables/speech_test_accuracy.npy"
)

epochs = range(
    1,
    len(train_acc)+1
)

plt.figure(figsize=(10,6))

plt.plot(

    epochs,

    train_acc,

    marker='o',

    label='Train Accuracy'
)

plt.plot(

    epochs,

    test_acc,

    marker='o',

    label='Test Accuracy'
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("REAL Speech Emotion Recognition Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(
    "Results/plots/real_accuracy_plot.png"
)

plt.show()

print("\nREAL accuracy plot generated!")