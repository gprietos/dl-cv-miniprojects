

# Implementation details
1) The image is resized with its shorter side randomly sampled in [256, 480] for scale augmentation
2) 224×224 crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted
standard color augmentation
batch normalization (BN) [16] right after each convolution and before activation
He initialization
SGD with a mini-batch size of 256
learning rate starts from 0.1 and is divided by 10 when the error plateaus
trained for up to 60 × 104 iterations
weight decay of 0.0001 and a momentum of 0.9
No dropout

standard 10-crop testing for comparison studies
fully-convolutional: average the scores at multiple scales (images are resized such that the shorter side is in {224, 256, 384, 480, 640}).


# Dataset 

- ImageNet 2012 Classification dataset.
  - 1000 Classes
  - train/val/test: 1.28M/50k/100k
  - top-1 and top-5 error rates
  - 224x224 img input size
  
- CIFAR-10 dataset
  - 10 Classes
  - train/test: 50k/10k
  - 32x32 img input size