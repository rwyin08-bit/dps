#!/usr/bin/env python3
"""
TEMPLATE for experiments 3-6.
Copy this file and replace the placeholder logic with actual implementation.

Each experiment should:
1. Load configuration from experiments.config
2. Run the specified computation
3. Save results as JSON to experiments/results/
4. Print progress and timing information
"""

import numpy as np
import json, os, time, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.config import *

def main():
    print(f"Experiment: TEMPLATE")
    print(f"Replace this with actual implementation.")
    print(f"See experiments/README.md for integration instructions.")

    # ── Example result structure ──
    result = {
        'experiment': 'template',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {},
        'note': 'Replace with actual computation',
    }

    outfile = os.path.join(RESULT_DIR, 'template_result.json')
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {outfile}")

if __name__ == '__main__':
    main()
