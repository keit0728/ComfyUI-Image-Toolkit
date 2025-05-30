import numpy as np
import torch


class AlphaFlatten:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Input images to flatten alpha channel into RGB colors."
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
                # RGB値を取得
                rgb = image[:, :, :3]
                # アルファ値を取得
                alpha = image[:, :, 3]
                
                # 白背景とのalpha合成処理
                # result = foreground_color * alpha + background_color * (1 - alpha)
                # 白背景：background_color = 1.0
                flattened_rgb = rgb * alpha[:, :, np.newaxis] + 1.0 * (1.0 - alpha[:, :, np.newaxis])
                
                result_images.append(flattened_rgb)
            else:
                # アルファチャンネルがない場合はそのまま返す
                result_images.append(image)

        result_np = np.stack(result_images)
        result_tensor = torch.from_numpy(result_np)

        if is_tensor and images.device.type != "cpu":
            result_tensor = result_tensor.to(images.device)

        return (result_tensor,)
