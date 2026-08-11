import torch
import matplotlib.pyplot as plt

def plot_loss_curves(history):
    loss = history['train_loss']
    val_loss = history['val_loss']
    accuracy = history['train_acc']
    val_accuracy = history['val_acc']
    epochs = range(len(history['train_loss']))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
