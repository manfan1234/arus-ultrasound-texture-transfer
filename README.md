# ARUS Ultrasound Texture Transfer

Code associated with the MSc thesis:

**Hyperrealistic Medical Image Simulations using Artificial Intelligence**

This repository contains PyTorch code for CycleGAN-based ultrasound image texture transfer between:

- a user-provided synthetic/source ultrasound image domain,
- and a user-provided real/target ultrasound image domain.

The repository provides two training modes:

1. **Baseline CycleGAN**  
   Standard non-residual image-to-image translation.

2. **Residual texture-transfer CycleGAN**  
   Residual generation of the form:

   `G(x) = x + alpha * R(x)`

   with additional losses for anatomical preservation and local texture matching.

## Important note

This repository does **not** include:

- the ARUS simulator,
- CUDA simulation kernels,
- CT/XCAT data,
- protected UCM code,
- clinical datasets,
- trained model weights.

Users must provide their own synthetic and real ultrasound image datasets.

## Repository structure

```text
src/
  models.py        Neural network architectures.
  datasets.py      Image loading and preprocessing.
  losses.py        GAN, gradient and texture losses.
  training.py      Generic CycleGAN training loop.
  inference.py     Inference utilities.
  utils.py         General utilities.

scripts/
  train_baseline.py   Train standard non-residual CycleGAN.
  train_residual.py   Train residual texture-transfer CycleGAN.
  run_inference.py    Apply a trained generator to a folder.
  smoke_test.py       End-to-end dummy test.
```

## Installation

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
Pillow
matplotlib
torch
torchvision
```

## Smoke test

Run a complete dummy end-to-end test:

```bash
python -m scripts.smoke_test
```

This creates synthetic toy images, trains for one batch, saves a checkpoint and runs inference.

Temporary files are written to:

```text
_smoke_test/
```

This folder is ignored by Git.

## Baseline training

The baseline model corresponds to a standard non-residual CycleGAN.

```bash
python -m scripts.train_baseline \
  --synthetic_dir /path/to/synthetic_images \
  --real_dir /path/to/real_images \
  --output_dir outputs/baseline_run \
  --epochs 20 \
  --img_size 256 \
  --batch_size 1 \
  --ngf 64 \
  --n_blocks 9 \
  --upsample_mode transpose
```

Useful options:

```text
--synthetic_dir       Folder containing source/synthetic images, domain X.
--real_dir            Folder containing target/real images, domain Y.
--output_dir          Folder where checkpoints are saved.
--epochs              Number of training epochs.
--img_size            Image size after preprocessing.
--batch_size          Batch size.
--ngf                 Number of generator filters in the first layer.
--n_blocks            Number of ResNet blocks in both generators.
--upsample_mode       transpose, bilinear or nearest.
--max_batches         Limit batches per epoch for quick tests.
--no_show_samples     Disable matplotlib sample display.
```

## Residual training

The residual model is designed to add realistic texture while preserving the source image structure.

The generator has the form:

```text
G(x) = x + alpha * R(x)
```

Example:

```bash
python -m scripts.train_residual \
  --synthetic_dir /path/to/synthetic_images \
  --real_dir /path/to/real_images \
  --output_dir outputs/residual_run \
  --epochs 20 \
  --img_size 256 \
  --batch_size 1 \
  --ngf 64 \
  --n_blocks_xy 9 \
  --n_blocks_yx 9 \
  --alpha 25 \
  --upsample_mode bilinear \
  --use_local_discriminator
```

Useful options:

```text
--alpha                       Residual scale in G(x) = x + alpha * R(x).
--n_blocks_xy                 Number of ResNet blocks in G_XY.
--n_blocks_yx                 Number of ResNet blocks in G_YX.
--lambda_cycle_x              Cycle-consistency weight for X reconstruction.
--lambda_cycle_y              Cycle-consistency weight for Y reconstruction.
--lambda_id_xy                Identity loss weight for G_XY(real_Y).
--lambda_id_yx                Identity loss weight for G_YX(real_X).
--lambda_direct_x             Direct preservation loss between fake_Y and real_X.
--lambda_gradient             Image-gradient preservation loss weight.
--lambda_high_freq            High-frequency energy matching loss weight.
--lambda_local_std            Local standard-deviation texture loss weight.
--use_local_discriminator     Enable additional local texture discriminator.
```

## Inference

Apply a trained generator to a folder of images.

Residual model example:

```bash
python -m scripts.run_inference \
  --model /path/to/generator_XY.pth \
  --input_dir /path/to/input_images \
  --output_dir outputs/translated \
  --ngf 64 \
  --n_blocks 9 \
  --upsample_mode bilinear \
  --residual \
  --alpha 25 \
  --img_size 256
```

Non-residual model example:

```bash
python -m scripts.run_inference \
  --model /path/to/generator_XY.pth \
  --input_dir /path/to/input_images \
  --output_dir outputs/translated \
  --ngf 64 \
  --n_blocks 9 \
  --upsample_mode transpose \
  --img_size 256
```

## Data format

Input folders should contain standard image files:

```text
.png
.jpg
.jpeg
.bmp
.tif
.tiff
```

Images are converted by default to grayscale and replicated over three channels before being passed to the network.

The default preprocessing is:

1. grayscale conversion with 3 output channels,
2. resize to `img_size x img_size`,
3. tensor conversion,
4. normalization to `[-1, 1]`.

## Checkpoints

Model checkpoints are saved as PyTorch `state_dict` files.

By default, model weights are ignored by Git:

```text
*.pth
*.pt
*.ckpt
```

Do not commit large model files directly to this repository.

## What is not included

This repository intentionally excludes all simulator-related and protected material:

```text
ARUS simulator
CUDA kernels
CT/XCAT data
NPZ volumes
clinical datasets
UCM-protected code
trained model weights
```

The code is intended to operate on image folders supplied by the user.

## Citation

If you use this code, please cite the associated MSc thesis:

```text
Manuel Muñoz Serna.
Hyperrealistic Medical Image Simulations using Artificial Intelligence.
MSc Thesis, Universidad Complutense de Madrid, 2026.
```

## License

To be defined.
