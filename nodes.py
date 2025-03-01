import numpy as np
import torch


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
                rgba_image[:, :, 3] = 1.0  # Range 0.0-1.0
            else:
                rgba_image = image.copy()

            # Set alpha values inversely proportional to brightness (bright=transparent, dark=opaque)
            brightness = np.mean(rgba_image[:, :, :3], axis=2)
            alpha = 1.0 - brightness
            rgba_image[:, :, 3] = alpha

            result_images.append(rgba_image)

        result_np = np.stack(result_images)
        result_tensor = torch.from_numpy(result_np)

        if is_tensor and images.device.type != "cpu":
            result_tensor = result_tensor.to(images.device)

        return (result_tensor,)


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
            grayscale = np.mean(image[:, :, :3], axis=2)
            binary_mask = np.where(grayscale >= normalized_threshold, 1.0, 0.0)

            for i in range(3):
                processed_image[:, :, i] = binary_mask

            result_images.append(processed_image)

        result_np = np.stack(result_images)
        result_tensor = torch.from_numpy(result_np)

        if is_tensor and images.device.type != "cpu":
            result_tensor = result_tensor.to(images.device)

        return (result_tensor,)


class GrayscaleImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Converts images to grayscale."},
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

        # 常に均等な重みを使用
        r_weight, g_weight, b_weight = 1 / 3, 1 / 3, 1 / 3

        for image in images_np:
            # グレースケール値を計算
            grayscale = (
                r_weight * image[:, :, 0]
                + g_weight * image[:, :, 1]
                + b_weight * image[:, :, 2]
            )

            # 出力画像を作成
            result_image = np.zeros_like(image)
            for i in range(3):
                result_image[:, :, i] = grayscale

            # アルファチャンネルがある場合は保持
            if image.shape[2] == 4:
                result_image[:, :, 3] = image[
                    :, :, 3
                ]  # アルファチャンネルをコピー

            result_images.append(result_image)

        result_np = np.stack(result_images)
        result_tensor = torch.from_numpy(result_np)

        if is_tensor and images.device.type != "cpu":
            result_tensor = result_tensor.to(images.device)

        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "BrightnessTransparency": BrightnessTransparency,
    "BinarizeImage": BinarizeImage,
    "GrayscaleImage": GrayscaleImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BrightnessTransparency": "BrightnessTransparency",
    "BinarizeImage": "BinarizeImage",
    "GrayscaleImage": "GrayscaleImage",
}
