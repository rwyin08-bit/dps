# Data Directory

## Benchmark Datasets

### SPE10 Comparative Solution Project
- Source: https://www.spe.org/web/csp/
- Reference: Christie & Blunt (2001), SPE Reservoir Evaluation & Engineering, 4(4), 308–317
- Canonical grid: 60 (NX) × 220 (NY) × 85 (NZ); we use 220×60 horizontal orientation (row-major transposition)
- Layers used: Layer 1 (2D, 13,200 cells), Layers 1–5 (3D, 66,000 cells)
- Wells: 5-spot pattern (1 injector + 4 producers)
- Flow: two-phase oil-water, incompressible

### Brugge Field
- Source: ISAPP Brugge Field Study
- Reference: Peters et al. (2010), SPE Reservoir Evaluation & Engineering, 13(3), 391–405
- Extended data: Peters et al. (2013), Computers & Geosciences, 50, 16–24
- Grid: 139 × 48 × 9 (~44,550 active cells)
- Wells: 20 producers + 10 injectors
- Flow: two-phase oil-water, undersaturated oil (no free gas)

## Generated Training Data

| Dataset | Contents | Size | Format |
|---------|----------|------|--------|
| SPE10 2D realizations | 10,000 geological realizations (MPS 5,000 + SGS 3,000 + PCA 2,000) | ~5 GB | .npy |
| SPE10 2D simulations | 5,000 MRST simulation input–output pairs | ~15 GB | .h5 |
| Brugge realizations | 5,104 geological realizations (104 official + 5,000 MPS) | ~8 GB | .npy |
| Brugge simulations | 5,000 MRST simulation input–output pairs | ~25 GB | .h5 |
| SBC calibration set | 200 independent prior realizations + observations (SPE10 2D) | ~500 MB | .npy |

All generated datasets are available from the corresponding author upon reasonable request due to their size (>50 GB total). Download instructions will be provided after manuscript acceptance.
