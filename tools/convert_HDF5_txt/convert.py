import argparse
import numpy as np
import h5py
import time
from memory_profiler import profile
from memory_profiler import memory_usage
import tracemalloc
import sys 
import os 

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)
from src.multiego.util.mat_modif_functions import monitor_performance


@monitor_performance()
def compress_to_hdf5(input_file, output_file):
    # Read the text file into a NumPy array
    data = np.loadtxt(input_file)

    # Save the data into an HDF5 file
    with h5py.File(output_file, 'w') as h5f:
        h5f.create_dataset('matrix', data=data, compression='gzip', compression_opts=9)

    print(f"Data successfully compressed and saved to {output_file}")

@monitor_performance()
def decompress_to_txt(input_file, output_file):
    # Read the HDF5 file
    with h5py.File(input_file, 'r') as h5f:
        data = h5f['matrix'][:]

    # Save the data into a text file
    np.savetxt(output_file, data, fmt='%g')

    print(f"Data successfully decompressed and saved to {output_file}")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Compress a text file into an HDF5 file or decompress an HDF5 file into a text file.")
    parser.add_argument("--input", required=True, help="Path to the input file.")
    parser.add_argument("--output", required=True, help="Path to the output file.")

    # Parse arguments
    args = parser.parse_args()

    # Determine operation based on file extension
    if not args.input.endswith('.h5'):
        compress_to_hdf5(args.input, args.output)
    elif args.input.endswith('.h5'):
        decompress_to_txt(args.input, args.output)
    else:
        raise ValueError("Unsupported file type. Input file must be either .txt or .h5")
