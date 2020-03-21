import numpy as np
from numpy.lib.stride_tricks import as_strided

__all__ = [
    'extract_patches',
    'CenterCrop', 'RandomCrop',
]

def extract_patches(img, patch_shape, strides=None):
    '''
    Divide numpy image into non-overlapped patches.
    Image tensor axes are arranged in form of [H(eight) W(idth) C(hannel)].
    '''
    if isinstance(patch_shape, int):
        patch_shape = [patch_shape] * 2
    strides = strides or patch_shape
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if len(img.shape) == 2:
        n_channel = 1
    else:
        n_channel = img.shape[-1]
    unit = img.strides[-1]
    cropped_shape = list(img.shape)
    cropped_shape[0] -= cropped_shape[0] % patch_shape[0]
    cropped_shape[1] -= cropped_shape[1] % patch_shape[1]
    # Draw 3D graph of the data, calculate step of jump.
    new_strides = (
        unit*n_channel*img.shape[1]*patch_shape[0], 
        unit*n_channel*patch_shape[1], 
        unit*n_channel*img.shape[1], 
        unit*n_channel, 
        unit,
    )
    new_shape = (
        cropped_shape[0] // patch_shape[0],
        cropped_shape[1] // patch_shape[1],
        patch_shape[0],
        patch_shape[1],
        n_channel,
    )
    return as_strided(img, shape=new_shape, strides=new_strides)


class Crop:
    NUMPY_IMAGE = (-3, -2)
    TORCH_IMAGE = (-2, -1)

    def __init__(self, crop_shape, crop_axes):
        if isinstance(crop_axes, int): crop_axes = (crop_axes,)
        if isinstance(crop_shape, int): crop_shape = (crop_shape,) * len(crop_axes)
        assert isinstance(crop_axes, (tuple, list)) and isinstance(crop_shape, (tuple, list)), 'Crop size and crop axes must be tuple, list or int.'
        self.crop_shape = np.array(crop_shape)
        self.crop_axes = np.array(crop_axes)

    def __call__(self, array):
        raise NotImplementedError


class CenterCrop(Crop):
    def __call__(self, array):
        """
        Center crop tensor-like data struct with shape of crop_axes
        along crop_size.

        array must at least implement __getitem__ and shape methods.
        """
        original_size = np.array(array.shape)[self.crop_axes]
        assert np.all(original_size >= self.crop_shape), 'Crop size must be smaller than original size.'
        float_ranges = original_size - self.crop_shape + 1
        positions = float_ranges // 2
        index = [slice(None)]*len(array.shape)
        for position, dim, length in zip(positions, self.crop_axes, self.crop_shape):
            index[dim] = slice(position, position + length)
        return array[tuple(index)]

class RandomCrop(Crop):

    def __call__(self, array):
        """
        Random crop tensor-like data struct with shape of crop_axes
        along crop_size.

        array must at least implement __getitem__ and shape methods.
        """
        original_size = np.array(array.shape)[self.crop_axes]
        float_ranges = original_size - self.crop_shape + 1
        ramdom_positions = np.random.randint(float_ranges)
        index = [slice(None)]*len(array.shape)
        for position, dim, length in zip(ramdom_positions, self.crop_axes, self.crop_shape):
            index[dim] = slice(position, position + length)
        return array[tuple(index)]