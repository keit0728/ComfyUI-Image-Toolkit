import numpy as np
import torch
from ..utils import convert_to_grayscale


class BinarizeImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to binarize."}),
                "threshold": (
                    "INT",
                    {
                        "default": 127,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "Values above threshold become white (1.0), below become black (0.0).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "ComfyUI-Image-Toolkit"

    def run(self, images, threshold):
        is_tensor = isinstance(images, torch.Tensor)

        if is_tensor:
            if images.device.type != "cpu":
                images = images.cpu()
            images_np = images.numpy()
        else:
            images_np = images

        result_images = []
        normalized_threshold = threshold / 255.0

        for image in images_np:
            processed_image = image.copy()
            grayscale = convert_to_grayscale(image)[:, :, 0]
            binary_mask = np.where(grayscale >= normalized_threshold, 1.0, 0.0)

            for i in range(3):
                processed_image[:, :, i] = binary_mask

            result_images.append(processed_image)

        result_np = np.stack(result_images)
        result_tensor = torch.from_numpy(result_np)

        if is_tensor and images.device.type != "cpu":
            result_tensor = result_tensor.to(images.device)

        return (result_tensor,)
