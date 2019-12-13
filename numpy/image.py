import numpy as np
from numpy.lib.stride_tricks import as_strided

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