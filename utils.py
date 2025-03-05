import numpy as np
import cv2


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    画像をグレースケールに変換します。
    OpenCVのcv2.cvtColorを使用して変換を行います。

    Args:
        image (np.ndarray): 入力画像（RGBまたはRGBA）

    Returns:
        np.ndarray: グレースケール化された画像（入力と同じチャンネル数）
    """
    # アルファチャンネルの有無を確認
    has_alpha = image.shape[2] == 4
    alpha = image[..., -1] if has_alpha else None

    # uint8に変換
    img_uint8 = (image * 255).astype(np.uint8)

    # RGBからBGRに変換
    if has_alpha:
        img_bgr = cv2.cvtColor(img_uint8[..., :3], cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    # グレースケールに変換
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 出力画像を作成
    result_image = np.zeros_like(image)
    for i in range(3):
        result_image[:, :, i] = gray.astype(np.float32) / 255.0

    # アルファチャンネルがある場合は保持
    if has_alpha:
        result_image[:, :, 3] = alpha

    return result_image
