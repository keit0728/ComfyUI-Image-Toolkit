import numpy as np
import torch
from PIL import Image


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
