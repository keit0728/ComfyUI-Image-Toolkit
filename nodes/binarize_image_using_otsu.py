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
            has_alpha = image.shape[-1] == 4
            # Store alpha channel if present
            alpha = image[..., -1] if has_alpha else None

            # Convert to grayscale if needed
            if image.shape[-1] == 3:
                # Convert to grayscale using our utility function
                gray = convert_to_grayscale(image)
            elif image.shape[-1] == 4:
                # For RGBA images, convert RGB part to grayscale
                gray = convert_to_grayscale(image[..., :3])
            else:
                gray = image

            # Ensure we have a single channel
            if len(gray.shape) > 2:
                gray = gray[..., 0]

            # Convert to uint8 for OpenCV
            gray_uint8 = (gray * 255).astype(np.uint8)

            # Apply Otsu's thresholding
            _, binary = cv2.threshold(
                gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Convert back to float32 and normalize
            binary_float = binary.astype(np.float32) / 255.0

            # Expand to 3 channels for RGB
            binary_float = np.stack([binary_float] * 3, axis=-1)

            # Restore alpha channel if it was present
            if has_alpha:
                binary_float = np.concatenate(
                    [binary_float, alpha[..., np.newaxis]], axis=-1
                )

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
