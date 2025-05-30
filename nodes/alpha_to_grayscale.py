import numpy as np
import torch


class AlphaToGrayscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Input images to convert alpha to grayscale."
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
            # アルファチャンネルがある場合のみ処理
            if image.shape[2] == 4:
                # アルファ値を取得
                alpha = image[:, :, 3]
                
                # アルファ値をグレースケール値に変換（BrightnessTransparencyの逆変換）
                # BrightnessTransparency: alpha = 1.0 - grayscale
                # AlphaToGrayscale: grayscale = 1.0 - alpha
                grayscale = 1.0 - alpha
                
                # RGB画像を作成（透明度なし）
                rgb_image = np.zeros(
                    (image.shape[0], image.shape[1], 3), dtype=np.float32
                )
                
                # 全チャンネルにグレースケール値を設定
                rgb_image[:, :, 0] = grayscale
                rgb_image[:, :, 1] = grayscale
                rgb_image[:, :, 2] = grayscale
                
                result_images.append(rgb_image)
            else:
                # アルファチャンネルがない場合はそのまま返す
                result_images.append(image)

        result_np = np.stack(result_images)
        result_tensor = torch.from_numpy(result_np)

        if is_tensor and images.device.type != "cpu":
            result_tensor = result_tensor.to(images.device)

        return (result_tensor,)

