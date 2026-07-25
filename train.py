from __future__ import annotations

import argparse

from training.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a brain tumor classifier.")
    parser.add_argument("--model", default="mobilenetv2", choices=["custom_cnn", "mobilenetv2", "efficientnetb0", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    args = parser.parse_args()
    train_model(args.model, args.epochs, args.batch_size, args.learning_rate, args.dropout)


if __name__ == "__main__":
    main()
