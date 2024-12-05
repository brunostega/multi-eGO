# Matrix Protonation State Adjustment Tool

This script is designed to adjust a matrix file to account for changes in the protonation state between two molecular topologies. It processes the input matrix using input and output topology files, ensuring consistency and correctness in the atom and molecule mappings.

---

## Usage

### Command-line Arguments
The script requires the following command-line arguments:

| Argument       | Type   | Required | Description                                                                                 |
|----------------|--------|----------|---------------------------------------------------------------------------------------------|
| `--input_mat`  | string | Yes      | Path to the input matrix file.                                                             |
| `--input_top`  | string | Yes      | Path to the input topology file.                                                           |
| `--output_top` | string | Yes      | Path to the output topology file.                                                          |
| `--out_name`   | string | No       | Name prefix for the output matrix file.                                                    |
| `--out`        | string | No       | Directory to save the output file. Default is the current directory (`.`).                 |

### Example Usage
```bash
python adjust_matrix.py \
    --input_mat path/to/matrix.ndx \
    --input_top path/to/input.top \
    --output_top path/to/output.top \
    --out_name adjusted_ \
    --out results/
```
