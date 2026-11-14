#!/bin/bash
# Fit the backbone CMAP of every residue listed on the command line, with the
# protocol used for the maps in cmaps/ (see cmaps/README.md).
#
#   GMXLIB=$PWD GMX=gmx tools/cmap_fit/fit_all.sh WORKDIR TARGETDIR ITER NS RES1 RES2 ...
#
# TARGETDIR holds the Choi & Pappu landscapes as <RESNAME>.csv.  The fitted
# map of each residue is copied to cmaps/<RESNAME>.npz as soon as it is done.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$1; TARGETS=$2; ITER=$3; NS=$4; shift 4
mkdir -p "$WORK"
for aa in "$@"; do
  dir="$WORK/$aa"
  python "$ROOT/tools/cmap_fit/make_dipeptide.py" "$aa" "$dir/input/$aa"
  python "$ROOT/multiego.py" --system "$aa" --egos mg --inputs_dir "$dir/input" --outputs_dir "$dir/out" \
         --explicit_name base --backend openmm > "$dir/generate.log" 2>&1
  python "$ROOT/tools/cmap_fit/fit_cmap.py" --params "$dir/out/$aa/base_1/mego_params.h5" \
         --conf "$dir/input/$aa/${aa}_box.gro" --target "$TARGETS/$aa.csv" --resname "$aa" \
         --output "$dir/${aa}_cmap.npz" --size 36 --iterations "$ITER" --ns "$NS" \
         --damping 0.9 --max-step 5 --average 3 --platform Reference > "$dir/fit.log" 2>&1
  python - "$dir/${aa}_cmap.npz" "$ROOT/cmaps/$aa.npz" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1])
np.savez_compressed(sys.argv[2], cmap=d["cmap"], target=d["target"], sampled=d["sampled"], temperature=d["temperature"])
PY
  echo "done $aa $(date +%T) $(grep '^final' "$dir/fit.log" | cut -c1-110)" >> "$WORK/progress.txt"
done
