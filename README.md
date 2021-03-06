# pytorch implementation for CartoonGAN and CycleGAN for avatar project
the CartoonGAN follows this [project](https://github.com/znxlwm/pytorch-CartoonGAN)\
the CycleGAN uses the vanilla cycleGAN [project](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) with some minor tweaks

### Folder structure
The following shows data folder structure.
```
── data
   ├── src_data # src data (not included in this repo)
   │   ├── train 
   │   └── test
   └── tgt_data # tgt data (not included in this repo)
       ├── train 
       └── pair # edge-promoting results to be saved here, only use for CartoonGAN

```
The training data needs to be downloaded from this link and place inside the data

## Usage for CartoonGAN
* Download VGG19
[VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
put the model under pre_trained_VGG19_model_path folder
* Run CartoonGAN.ipynb

## Usage for CycleGAN
* Run CycleGAN.ipynb

## Usage for Pix2Pix (Supervised GAN)
* Use the photo-manga pair images, the dataset can be downloaded from [here](https://drive.google.com/open?id=1nlx80wxkRyyc3h1lIz3p3M7DAiFIy4ef)
* Run Pix2Pix.ipynb

## Development Environment

* NVIDIA Tesla P100
* cuda 8.0
* python 3.5.5
* pytorch 0.4.1
* torchvision 0.2.1
* opencv-python 3.4.3.18

