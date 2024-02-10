## 3D-2D Neural Nets for Phase Retrieval in Noisy Interferometric Imaging

Uses 3D-to-2D UNet-style models to performs phase-image reconstructions from inputs consisting of noisy, randomly phase-modulated frames. Includes models for image reconstruction (3D-to-2D UNets, multiscale convolutional neural networks, and 3D-to-2D vision transformers) that either take the sampled frames directly, or operate on the correlation matrix of the frames. When compared with a singular value decomposition (SVD) approach, our models offer lower error and smoother reconstructions.

![Fig1_v7-01](https://github.com/andrewhproppe/PhaseRetrievalNNs/assets/68742471/0cd6940d-4c24-4835-8a79-6a70863c9132)

## Usage

Please see example scripts for training different models in scripts/model_training.

train_PRUNe.py: The main results of this work make use of the PRUNe (Phase Retreival U-Net) model, a 3D-2D convolutioanl autoencoder with both forward (e.g. standard ResNet block) and symmetric (U-Net-like) skip connections. We found that 3D convolutions over a 3D ensemble of noisy interferograms to encode a 2D latent space was the most performant model.

train_PRUNe2D.py Other models like PRUNe2D treat the 32 input interferograms as different convolutional channels instead, analogous to RGB channels of colored images. However, we found these 2D-2D U-Nets less accurate than our 3D-2D model.

## Contact

For any questions on this project, please contact Andrew Proppe (aproppe@uottawa.ca)

## Acknowledgements

We made use of the excellent [oxford_flowers102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102). We drew inspiration from the work of [X. Mao et al.](https://arxiv.org/pdf/1603.09056.pdf) using symmetric skip connections. A summary of related works can be found in the supporting information in our paper to which this work belongs.
