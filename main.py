import torch
import data_setup, model_builder, engine, utils

# Hyperparameters
NUM_EPOCHS = 4
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print("Modular Food Classifier Project Ready.")
    print("To run training, ensure dataset is available and call engine.train_step()")
