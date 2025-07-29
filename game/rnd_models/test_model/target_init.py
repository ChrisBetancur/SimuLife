#!/usr/bin/env python3
import sys
import os
import struct
import random

def overwrite_with_random_data(filename):
    """Overwrite binary file with random data while preserving structure."""
    print(f"\nProcessing file: {filename}")
    
    # Read original file header and data
    with open(filename, 'rb') as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError("Invalid header size")
        rows, cols = struct.unpack('<II', header)
        n_elements = rows * cols
        
        # Read the rest as float32 (4 bytes each)
        data_bytes = f.read(n_elements * 4)
        if len(data_bytes) < n_elements * 4:
            raise ValueError(f"Expected {n_elements} floats (4 bytes each), got {len(data_bytes)} bytes")
        
        print(f"Original dimensions: {rows} rows x {cols} columns")
        print("First 5 original float values:")
        for i in range(min(5, n_elements)):
            val = struct.unpack('<f', data_bytes[i*4:(i+1)*4])[0]
            print(f"  {val:.6f}")

    # Generate new random float32 data
    new_data = [random.uniform(-1.0, 1.0) for _ in range(n_elements)]
    
    # Overwrite file with header + new_data as float32
    with open(filename, 'wb') as f:
        f.write(header)
        for value in new_data:
            f.write(struct.pack('<f', value))
    
    # Verify
    with open(filename, 'rb') as f:
        f.read(8)
        updated = f.read(n_elements * 4)
        print("\nFirst 5 new float values:")
        for i in range(min(5, n_elements)):
            val = struct.unpack('<f', updated[i*4:(i+1)*4])[0]
            print(f"  {val:.6f}")
    
    return new_data, rows, cols

def decode_and_write_txt(bin_filename, txt_filename):
    """Decode binary file into text format."""
    with open(bin_filename, 'rb') as f:
        header = f.read(8)
        rows, cols = struct.unpack('<II', header)
        n_elements = rows * cols
        data_bytes = f.read(n_elements * 4)
        if len(data_bytes) != n_elements * 4:
            raise ValueError("File size doesn't match dimensions")
    
    with open(txt_filename, 'w') as out:
        out.write(f"File: {os.path.basename(bin_filename)}\n")
        out.write(f"Dimensions: {rows} x {cols}\n\n")
        for i in range(rows):
            row_vals = [
                f"{struct.unpack('<f', data_bytes[(i*cols + j)*4:(i*cols + j + 1)*4])[0]:.6f}"
                for j in range(cols)
            ]
            out.write(" ".join(row_vals) + "\n")
    print(f"\nDecoded text saved to {txt_filename}")

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base, 'target')
    for root, _, files in os.walk(target_dir):
        for fn in files:
            if fn == "nn_info.bin":
                print("Skipping nn_info.bin")
                continue
            if fn.endswith('.bin'):
                full = os.path.join(root, fn)
                overwrite_with_random_data(full)
                decode_and_write_txt(full, full.replace('.bin', '.txt'))
