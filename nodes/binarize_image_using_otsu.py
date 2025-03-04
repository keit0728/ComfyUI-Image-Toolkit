import numpy as np
import torch
import cv2
from ..utils import convert_to_grayscale


class BinarizeImageUsingOtsu:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to binarize."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "ComfyUI-Image-Toolkit"

    def run(self, images):
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
            # Convert to grayscale if needed
            if image.shape[-1] == 3:
                # Convert to grayscale using our utility function
                gray = convert_to_grayscale(image)[
                    :, :, 0
                ]  # Take only one channel
            else:
                gray = image

            # Convert to uint8 for OpenCV
            gray_uint8 = (gray * 255).astype(np.uint8)

            # Apply Otsu's thresholding
            _, binary = cv2.threshold(
                gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Convert back to float32 and normalize
            binary_float = binary.astype(np.float32) / 255.0

            # Expand to 3 channels for consistency
            binary_float = np.stack([binary_float] * 3, axis=-1)
            result_images.append(binary_float)

        # Stack the results back into a batch
        result_np = np.stack(result_images)

        # Convert back to tensor if input was tensor
        if is_tensor:
            result_tensor = torch.from_numpy(result_np)
            if images.device.type != "cpu":
                result_tensor = result_tensor.to(images.device)
            return (result_tensor,)

        return (result_np,)
