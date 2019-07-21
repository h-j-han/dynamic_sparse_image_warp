# Dynamic sparse_image_warp
This is sparse_image_warp module supporting dynamic shape tensor and I also provide time warp function for SpecAugment.

Current version of module 
[`sparse_image_warp`](https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/contrib/image/python/ops/sparse_image_warp.py) only support static shaped tensors. This repo is for upgraded sparse_image_warp.py to support computations of tensor with dynamic shape.

## Time warping in SpecAugment
In [specaugment paper](https://arxiv.org/abs/1904.08779), time warping is one of the three elements of augmentation techniques, and the module `sparse_image_warp` is used in time warping.

Here, I also provide `time_warp` funtion with dynamic supporting.

The configurations part of `time_warp` funtion is mainly inspired by 'Step 1 : Time warping' part in  [shelling203/SpecAugment](https://github.com/shelling203/SpecAugment/blob/master/SpecAugment/spec_augment_tensorflow.py)
