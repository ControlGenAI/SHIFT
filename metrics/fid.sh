#!/bin/bash
# clean-fid between two image folders (e.g. MS-COCO 5K real vs. generated).
# Usage: bash metrics/fid.sh <real_dir> <generated_dir>
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash metrics/fid.sh <real_dir> <generated_dir>" >&2
  exit 1
fi

python -c "
import sys
from cleanfid import fid
print(f'FID: {fid.compute_fid(sys.argv[1], sys.argv[2]):.2f}')
" "$1" "$2"
