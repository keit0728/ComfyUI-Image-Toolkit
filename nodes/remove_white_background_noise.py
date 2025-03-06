import cv2
import numpy as np
import torch
from ..utils import convert_to_grayscale


class RemoveWhiteBackgroundNoise:
    """
    白背景に混じったノイズを除去するノード
    指定した閾値で二値化を行い、白背景に置き換えます
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to process."}),
                "threshold": (
                    "INT",
                    {
                        "default": 240,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "Threshold value for binarization (0-255)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_noise"
    CATEGORY = "image/processing"

    def remove_noise(self, images, threshold):
        is_tensor = isinstance(images, torch.Tensor)

        if is_tensor:
            if images.device.type != "cpu":
                images = images.cpu()
            images_np = images.numpy()
        else:
            images_np = images

        # Process each image in the batch
        result_images = []
        for image in images_np:
            has_alpha = image.shape[-1] == 4
            # Store alpha channel if present
            alpha = image[..., -1] if has_alpha else None

            # グレースケール変換
            gray_image = convert_to_grayscale(image)

            # Convert to uint8 for OpenCV thresholding
            gray_uint8 = (gray_image[..., 0] * 255).astype(np.uint8)

            # Apply thresholding
            _, binary = cv2.threshold(
                gray_uint8, threshold, 255, cv2.THRESH_BINARY
            )

            # Create mask
            mask = binary == 255

            # Convert back to float32 and normalize
            img_float = image.astype(np.float32)

            # Apply white background
            img_float[mask] = [1.0, 1.0, 1.0]  # White in normalized RGB

            # Restore alpha channel if it was present
            if has_alpha:
                img_float = np.concatenate(
                    [img_float, alpha[..., np.newaxis]], axis=-1
                )

            result_images.append(img_float)

        # Stack the results back into a batch
        result_np = np.stack(result_images)

        # Convert back to tensor if input was tensor
        if is_tensor:
            result_tensor = torch.from_numpy(result_np)
            if images.device.type != "cpu":
                result_tensor = result_tensor.to(images.device)
            return (result_tensor,)

        return (result_np,)
