# I-CGAN
Torch implementation of Invertible -Conditional Generative Adversarial Networks on Celeb-A dataset . This is the implementation of the [IcGAN model proposed paper](https://arxiv.org/abs/1611.06355) .

The baseline used is the [Torch implementation](https://github.com/soumith/dcgan.torch) of the [DCGAN by Radford et al](http://arxiv.org/abs/1511.06434).

The architecture of the network is :

![image](/NET.png)

We will be training the model on Celeb A dataset .

You can also find the tensorflow implementation of IcGan here .

Application : Image Editing 

1 . RECONSTRUCTION OF IMAGES 

![image](/IMAGE.png)

2. SWAPPING OF ATTRIBUTES 

![image](/IMAGE1.png)

3. INTERPOLATION OF IMAGES

![image](/IMAGE2.png)


The IcGAN is trained in four steps. 

1. Train the generator. 
2. Create a dataset of generated images with the generator. 
3. Train the encoder Z to map an image *x* to a latent representation *z* with the dataset generated images. 
4. Train the encoder Y to map an image *x* to a conditional information vector *y* with the dataset of real images.

All the parameters of the training phase are located in cfg/mainConfig.lua.


### 1.1 Train with a face dataset: CelebA

Note: for speed purposes, the whole dataset will be loaded into RAM during training time, which requires about 10 GB of RAM. Therefore, 12 GB of RAM is a minimum requirement. Also, the dataset will be stored as a tensor to load it faster, make sure that you have around 25 GB of free space.

#### Preprocess
`mkdir celebA; cd celebA`

Download img_align_celeba.zip [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) under the link "Align&Cropped Images".
Also, you will need to download `list_attr_celeba.txt` from the same link, which is found under `Anno` folder. Create a folder called celebA and place your data their then unzip it . I would you suggest you unzip it using the terminal as it becomes faster .

```bash
unzip img_align_celeba.zip; cd ..
DATA_ROOT=celebA th data/preprocess_celebA.lua
```
Now move `list_attr_celeba.txt` to `celebA` folder.

```bash
mv list_attr_celeba.txt celebA
```


#### Training

* Conditional GAN: parameters are already configured to run CelebA (dataset=celebA, dataRoot=celebA).
	```bash
	th trainGAN.lua
	```

* Generate encoder dataset: 
	```bash
	net=[GENERATOR_PATH] outputFolder=celebA/genDataset/ samples=182638 th data/generateEncoderDataset.lua
	```
	(GENERATOR_PATH example: checkpoints/celebA_25_net_G.t7)

* Train encoder Z: 
	```
    datasetPath=celebA/genDataset/ type=Z th trainEncoder.lua
	```

* Train encoder Y: 
	```
    datasetPath=celebA/ type=Y th trainEncoder.lua
	```
## 2. Visualize the results

For visualizing the results you will need an already trained IcGAN (i.e. a generator and two encoders).
The parameters for generating results are in [`cfg/generateConfig.lua`](cfg/generateConfig.lua).

### 2.1 Reconstruct and modify real images

```bash
decNet=celeba_24_G.t7 encZnet=celeba_encZ_7.t7 encYnet=celeba_encY_5.t7 loadPath=[PATH_TO_REAL_IMAGES] th generation/reconstructWithVariations.lua
```
In my case loadPath = celebA/img_align_celeba/
### 2.2 Swap attributes

Swap the attribute information between two pairs of faces.

```bash
decNet=celeba_24_G.t7 encZnet=celeba_encZ_7.t7 encYnet=celeba_encY_5.t7 im1Path=[IM1] im2Path=[IM2] th generation/attributeTransfer.lua
```
### 2.3 Interpolate between faces

```bash
decNet=celeba_24_G.t7 encZnet=celeba_encZ_7.t7 encYnet=celeba_encY_5.t7 im1Path=[IM1] im2Path=[IM2] th generation/interpolate.lua
```
