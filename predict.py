from __future__ import annotations

import argparse

from prediction.predictor import predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict brain tumor class for a single MRI image.")
    parser.add_argument("image", help="Path to an MRI image.")
    args = parser.parse_args()
    result = predict_image(args.image)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    for label, probability in result["probabilities"].items():
        print(f"{label}: {probability:.2%}")


if __name__ == "__main__":
    main()
