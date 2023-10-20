Phase Retrieval Imaging with Intensity Correlations 

Uses 3D-to-2D UNet-style models to performs phase-image reconstructions from inputs consisting of noisy, randomly phase-modulated frames. Includes models for image reconstruction (3D-to-2D UNets, multiscale convolutional neural networks, and 3D-to-2D vision transformers) that either take the sampled frames directly, or operate on the correlation matrix of the frames. When compared with a singular value decomposition (SVD) approach, our models offer lower error and smoother reconstructions.

## Usage

## Changelog

- Transforms pipeline now work on dictionary batches for easier referencing
- Cleaned up dataset and datamodule implementation  in `g2_pcfs/pipeline/image_data.py`
- New model types within the `g2_pcfs.models.ode` module:
	- `g2_pcfs.models.ode.ode` contains the high level PyTorch Lightning
	- `g2_pcfs.models.ode.ode_models` contains implementations of models that slot in
- Working training script in `scripts/train_neuralode.py`
	- Nominally working, but adjoint training is very unstable and can stall because number of evaluations blow up

## TODO

- Test "real" architectures; MLP seems to work perfectly well but maybe we can improve on it
- Develop predict pipeline for inference
