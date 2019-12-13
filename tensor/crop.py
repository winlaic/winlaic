import numpy as np


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


