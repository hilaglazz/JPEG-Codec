import numpy as np
from PIL import Image

from utils import Q_C, Q_Y, dct2, pad_to_block, quantize, split_blocks, zigzag

EOB_TOKEN = -9999

def dc_predictive(blocks):
    # blocks: (num_blocks, 8, 8)
    # Flattens each block using a zigzag pattern and then uses predictive coding on the DC element (the first one)
    if len(blocks) == 0:
        return []
    
    # Convert blocks to zigzag format (num_blocks, 64)
    zz_blocks = np.array([zigzag(block) for block in blocks])
    
    # Apply DC predictive coding: DC[i] = DC[i] - DC[i-1]
    # First block uses previous DC = 0
    dc_values = zz_blocks[:, 0].copy()  # Extract all DC coefficients
    prev_dc = 0
    for i in range(len(dc_values)):
        current_dc = dc_values[i]
        zz_blocks[i, 0] = current_dc - prev_dc
        prev_dc = current_dc
    
    return zz_blocks

def rle_encode_block(zz):
    # zz: array or list of length 64
    # Encodes each sequence using Run-length encoding (RLE). Converts a sequence into a sequence of 
    # (number of trailing zeros, value) ending in (0, EOB_TOKEN). 
    # For example [0,0,11,0,0,0,12,0,13,0,0,0,0,0,...,0,0]--> [(2,11)(3,12)(1,13)(0,EOB_TOKEN)]
    if len(zz) != 64:
        raise ValueError("Input zz must be of length 64")
    
    # Convert to numpy array if needed
    zz = np.asarray(zz)
    
    # Find non-zero indices
    nonzero_indices = np.nonzero(zz)[0]
    
    if len(nonzero_indices) == 0:
        # All zeros - just return EOB token
        return [(0, EOB_TOKEN)]
    
    result = []
    prev_idx = -1
    
    # Process each non-zero value
    for idx in nonzero_indices:
        run_length = idx - prev_idx - 1
        result.append((run_length, int(zz[idx])))
        prev_idx = idx
    
    # Append EOB token
    result.append((0, EOB_TOKEN))
    
    return result


def rle_encode_blocks(blocks):
    # blocks: (num_blocks, 8, 8)
    # Apply DC predictive coding before RLE
    zz_blocks = dc_predictive(blocks)
    
    if len(zz_blocks) == 0:
        return np.array([], dtype=[('run', np.int16), ('val', np.int16)])
    
    # Encode all blocks
    rle_stream = []
    for zz in zz_blocks:
        rle_stream.extend(rle_encode_block(zz))
    
    # Convert to numpy array with specified dtype for efficient storage
    return np.array(rle_stream, dtype=[('run', np.int16), ('val', np.int16)])


def compress_image(input_path, output_path, qscale=1.0):
    """
    Compress an image using JPEG-like steps and save to output_path.
    Steps:
    1. Load image, convert to YCbCr, and save Y, Cb, Cr channels to a compressed file.
    2. Chroma subsampling 
    3. Split into 8x8 blocks (with padding if needed)
    4. Apply DCT to each block
    5. Quantize DCT coefficients
    6. Zigzag order
    7. Serialize and compress (np.savez_compressed)
    
    Args:
        input_path: Path to input image file
        output_path: Path to save compressed .npz file
        qscale: Scalar to multiply quantization matrices (default 1.0, higher = more compression)
    """
    if qscale <= 0:
        raise ValueError("qscale must be positive")
    
    img = Image.open(input_path).convert('YCbCr')
    Y_img, Cb_img, Cr_img = img.split()
    w, h = Y_img.size

    # Downsample using PIL resize with bicubic interpolation
    Cb_sub = Cb_img.resize((w//2, h//2), Image.BICUBIC)
    Cr_sub = Cr_img.resize((w//2, h//2), Image.BICUBIC)

    # Convert to arrays
    Y = np.array(Y_img)
    Cb_sub = np.array(Cb_sub)
    Cr_sub = np.array(Cr_sub)

    # Pad and split into 8x8 blocks
    Y_pad, Y_shape = pad_to_block(Y)
    Cb_pad, Cb_shape = pad_to_block(Cb_sub)
    Cr_pad, Cr_shape = pad_to_block(Cr_sub)
    Y_blocks = split_blocks(Y_pad)
    Cb_blocks = split_blocks(Cb_pad)
    Cr_blocks = split_blocks(Cr_pad)

    # Apply DCT and quantization to each block
    # Shift by -128 first to center the values around 0 for DCT
    # Process Y channel
    Y_centered = Y_blocks.astype(np.float32) - 128.0
    Y_dct = np.array([dct2(block) for block in Y_centered])
    Y_quant = np.array([quantize(block, Q_Y * qscale) for block in Y_dct])
    
    # Process Cb channel
    Cb_centered = Cb_blocks.astype(np.float32) - 128.0
    Cb_dct = np.array([dct2(block) for block in Cb_centered])
    Cb_quant = np.array([quantize(block, Q_C * qscale) for block in Cb_dct])
    
    # Process Cr channel
    Cr_centered = Cr_blocks.astype(np.float32) - 128.0
    Cr_dct = np.array([dct2(block) for block in Cr_centered])
    Cr_quant = np.array([quantize(block, Q_C * qscale) for block in Cr_dct])

    # Convert quantized blocks to int16 to ensure compatibility with RLE encoding
    # (quantize returns int8, but we need int16 for RLE to handle larger coefficients)
    Y_quant = Y_quant.astype(np.int16)
    Cb_quant = Cb_quant.astype(np.int16)
    Cr_quant = Cr_quant.astype(np.int16)
    
    # Apply RLE encoding (with DC predictive coding inside)
    Y_rle = rle_encode_blocks(Y_quant)
    Cb_rle = rle_encode_blocks(Cb_quant)
    Cr_rle = rle_encode_blocks(Cr_quant)

    np.savez_compressed(
        output_path,
        Y_rle=Y_rle,
        Cb_rle=Cb_rle,
        Cr_rle=Cr_rle,
        Y_blocks=len(Y_blocks),
        Cb_blocks=len(Cb_blocks),
        Cr_blocks=len(Cr_blocks),
        Y_shape=Y_shape,
        Cb_shape=Cb_shape,
        Cr_shape=Cr_shape,
        qscale=qscale
    ) 
