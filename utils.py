import numpy as np
from scipy.fft import dctn, idctn


# Standard JPEG quantization matrices
Q_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)

Q_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)

# DCT and inverse DCT helpers
def dct2(block):
    return dctn(block, type=2, norm='ortho')

def idct2(block):
    return idctn(block, type=2, norm='ortho')

# Quantization and dequantization helpers
def quantize(block, q_matrix):
    # Quantize a block using a q_matrix values. Returns in np.int8 format
    # TODO: implement this
    # round to nearest integer and convert to int8
    return np.round(block / q_matrix).astype(np.int8)


def dequantize(block, q_matrix):
    return (block * q_matrix).astype(np.float32)

# Zigzag and inverse zigzag helpers
_zigzag_indices = np.array([
    [ 0,  1,  5,  6, 14, 15, 27, 28],
    [ 2,  4,  7, 13, 16, 26, 29, 42],
    [ 3,  8, 12, 17, 25, 30, 41, 43],
    [ 9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63]
]).flatten()

def zigzag(block):
    return block.flatten()[_zigzag_indices]

def inverse_zigzag(arr):
    out = np.zeros(64, dtype=arr.dtype)
    out[_zigzag_indices] = arr
    return out.reshape(8, 8)

# Block splitting and merging helpers
def pad_to_block(arr, block_size=8):
    h, w = arr.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='constant')
    return padded, arr.shape

def split_blocks(arr, block_size=8):
    h, w = arr.shape
    blocks = arr.reshape(h // block_size, block_size, w // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)
    return blocks

def merge_blocks(blocks, image_shape, block_size=8):
    h, w = image_shape
    h_padded = ((h + block_size - 1) // block_size) * block_size
    w_padded = ((w + block_size - 1) // block_size) * block_size
    blocks_per_row = w_padded // block_size
    blocks_per_col = h_padded // block_size
    arr = blocks.reshape(blocks_per_col, blocks_per_row, block_size, block_size)
    arr = arr.transpose(0, 2, 1, 3).reshape(h_padded, w_padded)
    return arr[:h, :w]
