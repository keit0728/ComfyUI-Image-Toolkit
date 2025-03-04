import numpy as np
import torch
from PIL import Image
from .utils import convert_to_grayscale


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

        for image in images_np:
            result_image = convert_to_grayscale(image)
            result_images.append(result_image)

        result_np = np.stack(result_images)
        result_tensor = torch.from_numpy(result_np)

        if is_tensor and images.device.type != "cpu":
            result_tensor = result_tensor.to(images.device)

        return (result_tensor,)


class AntialiasingImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Images to apply antialiasing effect."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "ComfyUI-Image-Toolkit"

    def run(self, images):
        is_tensor = isinstance(images, torch.Tensor)
        device = images.device if is_tensor else None

        # 画像をCPUに移動し、NumPy配列に変換
        if is_tensor:
            if images.device.type != "cpu":
                images = images.cpu()
            images_np = images.numpy()
        else:
            images_np = images

        # LANCZOSフィルターを固定で使用
        pil_filter = (
            Image.LANCZOS
            if hasattr(Image, "LANCZOS")
            else Image.Resampling.LANCZOS
        )

        result_images = []
        # 固定のスケール倍率を2.0に設定
        scale_factor = 2.0

        for image in images_np:
            # NumPy配列をPIL画像に変換（0-1の範囲を0-255に変換）
            pil_image = Image.fromarray(
                (image[:, :, :3] * 255).astype(np.uint8)
            )

            # 元のサイズを取得
            original_size = pil_image.size

            # 固定倍率2.0に拡大
            upscaled = pil_image.resize(
                (
                    int(original_size[0] * scale_factor),
                    int(original_size[1] * scale_factor),
                ),
                pil_filter,
            )

            # 元のサイズに縮小（アンチエイリアス効果）
            processed = upscaled.resize(original_size, pil_filter)

            # PIL画像をNumPy配列に戻す（0-255の範囲を0-1に変換）
            processed_np = np.array(processed).astype(np.float32) / 255.0

            # 結果画像の作成
            result_image = np.zeros_like(image)
            result_image[:, :, :3] = processed_np

            # アルファチャンネルがある場合は保持
            if image.shape[2] == 4:
                result_image[:, :, 3] = image[:, :, 3]

            result_images.append(result_image)

        # 処理結果をスタックしてテンソルに変換
        result_np = np.stack(result_images)
        result_tensor = torch.from_numpy(result_np)

        # 元のデバイスに戻す
        if is_tensor and device and device.type != "cpu":
            result_tensor = result_tensor.to(device)

        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "BrightnessTransparency": BrightnessTransparency,
    "BinarizeImage": BinarizeImage,
    "GrayscaleImage": GrayscaleImage,
    "AntialiasingImage": AntialiasingImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BrightnessTransparency": "BrightnessTransparency",
    "BinarizeImage": "BinarizeImage",
    "GrayscaleImage": "GrayscaleImage",
    "AntialiasingImage": "AntialiasingImage",
}
