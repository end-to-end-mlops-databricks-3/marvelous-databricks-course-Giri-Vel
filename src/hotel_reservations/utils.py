"""Utility class."""

import numpy as np

# def adjust_predictions(predictions: np.ndarray, scale_factor: float = 1.3) -> np.ndarray:
#     """Adjust predictions by multiplying them with a scale factor.

#     :param predictions: Array of predictions to be adjusted
#     :param scale_factor: Factor to scale the predictions by
#     :return: Adjusted predictions array
#     """

#     return [round(pred * scale_factor, 2) for pred in predictions]


def adjust_predictions(predictions: np.ndarray, scale_factor: float = 1.3) -> np.ndarray:
    """Adjust predictions by multiplying them with a scale factor.

    :param predictions: Array of predictions to be adjusted
    :param scale_factor: Factor to scale the predictions by
    :return: Adjusted predictions array
    """
    # If predictions are categorical (e.g., "not_canceled"/"canceled"), map them accordingly
    mapping = {"Not_Canceled": "Room Reserved", "Canceled": "Room Not Reserved"}
    try:
        # return [mapping[pred] if pred in mapping else pred for pred in predictions]
        return [mapping.get(pred, pred) for pred in predictions]
    except Exception:
        return ["not able to decode the string" for _ in predictions]
