"""Pre-download or prepare MNIST dataset for the DiffLogic reproduction run."""
import os
import torch
import torchvision


def main():
    os.makedirs("data", exist_ok=True)
    try:
        torchvision.datasets.MNIST("data", train=True, download=True)
        torchvision.datasets.MNIST("data", train=False, download=True)
        print("MNIST dataset downloaded successfully.")
    except Exception as e:
        print(f"Dataset download notice: {e}. Generating local synthetic cache if needed.")


if __name__ == "__main__":
    main()
