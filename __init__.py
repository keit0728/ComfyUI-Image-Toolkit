from .nodes.grayscale_image import GrayscaleImage
from .nodes.brightness_transparency import BrightnessTransparency
from .nodes.binarize_image import BinarizeImage
from .nodes.antialiasing_image import AntialiasingImage

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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
