import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    画像をグレースケールに変換します。

    Args:
        image (np.ndarray): 入力画像（RGBまたはRGBA）

    Returns:
        np.ndarray: グレースケール化された画像（入力と同じチャンネル数）
    """
    # ITU-R BT.601規格に基づく輝度変換の重み
    # 人間の目は緑に最も敏感で、青に最も鈍感なため、緑の重みが最大になっている
    r_weight, g_weight, b_weight = 0.299, 0.587, 0.114

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
        result_image[:, :, 3] = image[:, :, 3]

    return result_image
