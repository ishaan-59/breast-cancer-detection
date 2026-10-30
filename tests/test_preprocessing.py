import numpy as np
import pytest

from breast_cancer_detection.preprocessing import prepare_image_tensor


def test_prepare_image_tensor_shape() -> None:
    image = np.zeros((1024, 640), dtype=np.uint16)
    tensor = prepare_image_tensor(image)
    assert tuple(tensor.shape) == (1, 3, 1024, 640)


def test_prepare_image_tensor_rejects_wrong_dtype() -> None:
    image = np.zeros((1024, 640), dtype=np.uint8)
    with pytest.raises(TypeError):
        prepare_image_tensor(image)
