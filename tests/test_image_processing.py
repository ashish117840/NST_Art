import unittest
from PIL import Image

from app import resize_image_for_model


class ResizeImageForModelTests(unittest.TestCase):
    def test_resizes_to_square_target_size(self):
        image = Image.new("RGB", (100, 200), color=(255, 0, 0))

        resized = resize_image_for_model(image, target_size=512)

        self.assertEqual(resized.size, (512, 512))
        self.assertEqual(resized.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
