import numpy as np
import torch
from ..utils import convert_to_grayscale


class BrightnessTransparency:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Input images to adjust transparency. Bright areas will become transparent."
                    },
                ),
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

        result_images = []

        for image in images_np:
            if image.shape[2] == 3:
                rgba_image = np.zeros(
                    (image.shape[0], image.shape[1], 4), dtype=np.float32
                )
                rgba_image[:, :, :3] = image
                rgba_image[:, :, 3] = 1.0
            else:
                rgba_image = image.copy()

            grayscale_image = convert_to_grayscale(rgba_image)
            grayscale = grayscale_image[:, :, 0]  # グレースケール値取得
            alpha = 1.0 - grayscale
            rgba_image[:, :, 3] = alpha

            result_images.append(rgba_image)

        result_np = np.stack(result_images)
        result_tensor = torch.from_numpy(result_np)

        if is_tensor and images.device.type != "cpu":
            result_tensor = result_tensor.to(images.device)

        return (result_tensor,)
