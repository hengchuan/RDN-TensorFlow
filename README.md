# RDN-TensorFlow
A TensorFlow implementation of CVPR 2018 paper [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797).  
Official implementation: [Torch code for our CVPR 2018 paper "Residual Dense Network for Image Super-Resolution" (Spotlight)](https://github.com/yulunzhang/RDN)
  
## Prerequisites
- Python-2.7(Python-3.5 for branch python3)
- TensorFlow-1.10.0
- Numpy-1.14.5
- OpenCV-2.4.9.1
- h5py-2.6.0
  
## Usage
### Prepare data
Download DIV2K training data from [here](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip).  
Extract and place all the images in RDN-TensorFlow/Train/DIV2K_train_HR.
### Train
`python main.py`
### Test
`python main.py --is_train=False`


If you want to use the resize function in MATLAB when generating training data as the pretrained model used, you need to install [MATLAB API for Python](http://www.mathworks.com/help/matlab/matlab-engine-for-python.html), and run the script with option `--matlab_bicubic=True`.
### Train
`python main.py --matlab_bicubic=True`
### Test
`python main.py --is_train=False --matlab_bicubic=True`
  
## References
- [kweisamx/TensorFlow-ESPCN](https://github.com/kweisamx/TensorFlow-ESPCN)
