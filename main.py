"""
presented by:
Michal Shoob- 211534953
Hila Glazz- 207881756
"""

import os
import numpy as np
from PIL import Image
from compressor import compress_image
from decompressor import decompress_image
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qscale', type=float, default=1.0, help='Quantization scale (default: 1.0)')
    args = parser.parse_args()
    qscale = args.qscale

    INPUT_IMAGE = 'crab_nabula.tif'
    COMPRESSED_FILE = f'compressed_q{qscale:.2f}.npz'
    OUTPUT_IMAGE = f'output_q{qscale:.2f}.tif'

    # Step 1: Compress
    compress_image(INPUT_IMAGE, COMPRESSED_FILE, qscale=qscale)
    print(f"Compressed file saved as {COMPRESSED_FILE}")

    # Calculate compression ratio
    orig_size = os.path.getsize(INPUT_IMAGE)
    comp_size = os.path.getsize(COMPRESSED_FILE)
    ratio = orig_size / comp_size if comp_size > 0 else float('inf')
    print(f"Compression ratio: {ratio:.2f}")

    # Step 2: Decompress
    decompress_image(COMPRESSED_FILE, OUTPUT_IMAGE)
    print(f"Decompressed image saved as {OUTPUT_IMAGE}")

    # Step 3: Compare decompressed to original
    orig = Image.open(INPUT_IMAGE).convert('RGB')
    dec = Image.open(OUTPUT_IMAGE).convert('RGB')
    orig_np = np.array(orig).astype(np.float32)
    dec_np = np.array(dec).astype(np.float32)

    # Compute relative Euclidean distance
    numerator = np.linalg.norm(orig_np - dec_np)
    denominator = np.linalg.norm(orig_np)
    rel_dist = numerator / denominator if denominator != 0 else float('inf')
    print(f"Relative Euclidean distance between original and decompressed: {rel_dist:.6f}") 