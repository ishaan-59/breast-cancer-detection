import numpy as np

from scripts.build_model_ready_cache import resize_and_pad_pair


def test_resize_and_pad_pair_contract() -> None:
    image = np.full((8, 4), 123, dtype=np.uint16)
    mask = np.zeros((8, 4), dtype=np.uint8)
    mask[2:6, 1:3] = 255
    output_image, output_mask = resize_and_pad_pair(image, mask)
    assert output_image.shape == (1024, 640)
    assert output_image.dtype == np.uint16
    assert output_mask.shape == (1024, 640)
    assert output_mask.dtype == np.uint8
    assert set(np.unique(output_mask)).issubset({0, 255})
