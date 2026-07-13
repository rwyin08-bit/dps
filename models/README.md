# Pre-trained Model Weights

Pre-trained weights for all DPS-HM components are available upon request.

## Available Models

| Component | File | Size | Description |
|-----------|------|------|-------------|
| VAE | vae_spe10_2d.pt | ~150 MB | Encoder/decoder, d_z=256, β=0.5 |
| VAE | vae_spe10_3d.pt | ~200 MB | Encoder/decoder, d_z=512 |
| VAE | vae_brugge.pt | ~250 MB | Encoder/decoder for Brugge |
| LDM | ldm_spe10_2d.pt | ~1.1 GB | Score network s_θ, ~280M params, cosine β(t) |
| LDM | ldm_spe10_3d.pt | ~1.1 GB | Score network for 3D |
| LDM | ldm_brugge.pt | ~1.1 GB | Score network for Brugge |
| FNO | fno_spe10_2d.pt | ~300 MB | U-FNO surrogate, 16 modes/dim, 4 Fourier layers |
| FNO | fno_spe10_3d.pt | ~350 MB | U-FNO surrogate for 3D |
| FNO | fno_brugge.pt | ~400 MB | U-FNO surrogate for Brugge |

## Training Configuration

All models trained with fixed random seeds for reproducibility:
- VAE: seed=42 (Adam, lr=1e-4, batch=64, 200 epochs)
- LDM: seed=123 (AdamW, lr=1e-4, batch=32, 500k steps)
- FNO: seed=456 (Adam, lr=1e-3, batch=16, 100 epochs)

Hardware: Single NVIDIA A100 (40 GB), PyTorch 2.1, CUDA 12.1

## Download

Model weights will be deposited at a public repository (Zenodo/Hugging Face) upon manuscript acceptance. For early access, contact the corresponding author: swzhang189@163.com.
