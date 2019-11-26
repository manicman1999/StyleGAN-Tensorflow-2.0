# StyleGAN Tensorflow 2.0
Unofficial implementation of StyleGAN using TensorFlow 2.0.
Original paper: https://arxiv.org/abs/1812.04948

This implementation does not include functionality for Growing GAN, but will soon include Multi-Scale gradient capabilities.
See this paper for more details: https://arxiv.org/abs/1903.06048


## Image Samples
Trained on Landscapes:

![Teaser image](./landscapes.png)


Mixing Styles:

![Teaser image](./styles.png)


## Web Demo
A web demo for generating your own landscapes live:

https://matchue.ca/p/earthhd/


## Before Running
Please ensure you have created the following folders:
1. /Models/
2. /Results/
3. /data/

Additionally, please ensure that your folder with images is in /data/ and changed at the top of stylegan.py.
