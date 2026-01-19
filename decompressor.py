import numpy as np
from PIL import Image

from utils import Q_C, Q_Y, dequantize, idct2, inverse_zigzag, merge_blocks

EOB_TOKEN = -9999

def rle_decode_block(rle_pairs):
    # rle_pairs: a list [(zeros, val)] to convert back to an array of size 64
    # For example [(2,11)(3,12)(1,13)(0,EOB_TOKEN)] --> [0,0,11,0,0,0,12,0,13,0,0,0,0,0,...,0,0]
    zz = np.zeros(64, dtype=np.int16)
    pos = 0
    
    # Iterate through the RLE pairs and reconstruct the zigzag array
    for run, val in rle_pairs:
        # If we hit the end of block token, we stop processing
        if val == EOB_TOKEN:
            break
        # Skip zeros for the run length
        pos += run
        # Ensure we don't exceed array bounds
        if pos < 64:
            zz[pos] = val
            pos += 1
    
    return zz


def dc_predictive_decode(zz_blocks):
    # zz_blocks: given an array of blocks in zigzag format (num_blocks, 64) decode the predictive coding on the first element of each block
    if len(zz_blocks) == 0:
        return zz_blocks
    
    # Ensure zz_blocks is a numpy array
    zz_blocks = np.asarray(zz_blocks, dtype=np.int16)
    
    # Decode DC predictive coding: DC[i] = DC_diff[i] + DC[i-1]
    # First block uses previous DC = 0
    prev_dc = 0
    for i in range(len(zz_blocks)):
        # Decode the current DC coefficient by adding the previous DC value
        zz_blocks[i, 0] = zz_blocks[i, 0] + prev_dc
        prev_dc = zz_blocks[i, 0]  # Update the previous DC value for the next iteration
    
    return zz_blocks


def rle_decode_stream(rle_stream, num_blocks):
    # rle_stream: an array of num_blocks blocks each encoded in RLE coding and concatenated.
    # Need to separate into a list of RLE encoded blocks, decode each one, decode the predictive coding and return an array of 8x8 blocks
    if num_blocks == 0:
        return np.array([]).reshape(0, 8, 8)
    
    zz_blocks = []
    i = 0
    stream_len = len(rle_stream)
    
    # Each block is separated by an EOB_TOKEN, so we need to iterate through the stream
    for block_idx in range(num_blocks):
        current_block = []
        # Collect RLE pairs until we hit EOB_TOKEN
        while i < stream_len:
            run, val = rle_stream[i]  # Get the next (run, value) pair
            i += 1  # Move to the next pair
            current_block.append((run, val))
            # If we hit the end of block token, we stop collecting for this block
            if val == EOB_TOKEN:
                break
        
        # Decode the current block from RLE
        zz = rle_decode_block(current_block)
        zz_blocks.append(zz)
    
    # Convert to numpy array (num_blocks, 64)
    zz_blocks = np.array(zz_blocks, dtype=np.int16)
    
    # DC predictive decode
    zz_blocks = dc_predictive_decode(zz_blocks)
    
    # Inverse zigzag to convert back to 8x8 blocks
    blocks = np.array([inverse_zigzag(zz) for zz in zz_blocks])
    
    return blocks

def decompress_image(input_path, output_path):
    """
    Decompress a JPEG-like compressed file and save as an image.
    Always reads qscale from the file.
    
    Args:
        input_path: Path to compressed .npz file
        output_path: Path to save decompressed image
    """
    try:
        data = np.load(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Compressed file not found: {input_path}")
    except Exception as e:
        raise ValueError(f"Error loading compressed file: {e}")
    
    # Extract data from compressed file
    Y_rle = data['Y_rle']
    Cb_rle = data['Cb_rle']
    Cr_rle = data['Cr_rle']
    Y_blocks = int(data['Y_blocks'])
    Cb_blocks = int(data['Cb_blocks'])
    Cr_blocks = int(data['Cr_blocks'])
    Y_shape = tuple(data['Y_shape'])
    Cb_shape = tuple(data['Cb_shape'])
    Cr_shape = tuple(data['Cr_shape'])
    qscale = float(data['qscale']) if 'qscale' in data else 1.0

    # Decode RLE streams (with DC predictive decoding)
    Y_blocks_arr = rle_decode_stream(Y_rle, Y_blocks)
    Cb_blocks_arr = rle_decode_stream(Cb_rle, Cb_blocks)
    Cr_blocks_arr = rle_decode_stream(Cr_rle, Cr_blocks)

    # Dequantize each block using the quantization matrices
    Y_deq = np.array([dequantize(block, Q_Y * qscale) for block in Y_blocks_arr])
    Cb_deq = np.array([dequantize(block, Q_C * qscale) for block in Cb_blocks_arr])
    Cr_deq = np.array([dequantize(block, Q_C * qscale) for block in Cr_blocks_arr])

    # Apply inverse DCT to each block, shift by +128, and clip to valid pixel range
    Y_blocks_arr = np.array([np.clip(idct2(block) + 128, 0, 255) for block in Y_deq])
    Cb_blocks_arr = np.array([np.clip(idct2(block) + 128, 0, 255) for block in Cb_deq])
    Cr_blocks_arr = np.array([np.clip(idct2(block) + 128, 0, 255) for block in Cr_deq])

    # Merge blocks and crop to original shape
    Y = merge_blocks(Y_blocks_arr, Y_shape)
    Cb = merge_blocks(Cb_blocks_arr, Cb_shape)
    Cr = merge_blocks(Cr_blocks_arr, Cr_shape)

    h, w = Y_shape
    # Upsample Cb and Cr to full size using PIL resize (bicubic interpolation)
    Cb_img = Image.fromarray(Cb.astype(np.uint8), mode='L')
    Cr_img = Image.fromarray(Cr.astype(np.uint8), mode='L')
    Cb_up = np.array(Cb_img.resize((w, h), Image.BICUBIC))
    Cr_up = np.array(Cr_img.resize((w, h), Image.BICUBIC))

    # Combine channels and convert back to RGB
    ycbcr = np.stack([Y, Cb_up, Cr_up], axis=2).astype(np.uint8)
    img = Image.fromarray(ycbcr, mode='YCbCr').convert('RGB')
    img.save(output_path) 
