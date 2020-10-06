# Command line interface documentation

## Training
```
yapic train <network> <image_path> <label_path> [options]
```

## Prediction
```
yapic predict <network> <image_path> <output_path> [options]
```

## Deployment: Export to DeepImageJ
```
yapic deploy <network> <image_path> <output_path> [options]
```


## Parameters

### network

Either a model file in h5 format to use a pretrained model or specific string to initialize a new model.

* Choose ```unet_2d``` or ```unet_multi_z``` to initialize a new model.
    * ```unet_2d```: The original U-Net network as described by
      [Ronneberger et al.](https://arxiv.org/pdf/1505.04597.pdf) with
      zxy size of 1x572x572. You can train 2D images as well as 3D multichannel data with this model (e.g. z-stacks asquired with a confocal microscope). However, the model will be trained with
      single 2D slices of your 3D data.
    * ```unet_multi_z```: Combination of 5 ```unet_2d``` models to  
      process 3D data. It takes 5 z-slices as input to predict the
      slice in the middle.

* Use ```path/to/my/pretrained_model.h5``` to continue training of a    
  pretrained keras model.


### image_path

Define a folder with *tiff* or *tif* images
```
path/to/my/images
```

or a wildcard
```
"path/to/my/images/*.tif"
```

Don't forget double quotes in case of wildcards!

##### Input image format
YAPic supports *tif* and *tiff* files
* RGB images
* Multichannel images
* Z-stacks

Especially in case of multidimensional images:  Make sure to always
convert your pixel images with [Fiji](https://fiji.sc) before using YAPiC.
Large amounts of image data can be conveniently converted with Fiji by using
[batch processing](https://imagej.net/Batch_Processing).

### label_path

Define a path to an *Ilastik Project File (.ilp)*
```
path/to/my/ilastik_project.ilp
```

or to label masks in *tif* format.
```
path/to/my/labelfiles/
```
```
"path/to/my/labelsfiles/*.tif"
```

##### Ilastik Project Files
The images in associated with your Ilastik project have to be identical with
the *tif* images you define in the *image_path* argument.

##### Label masks in *tif* format

* The label image have to have identical dimension in z, x and y as the corresponding
  pixel images. They always have one channel.
  Pixel integer values define the class labels:
  * 0: no label
  * 1: class 1
  * 2: class 2
  * 3: class 3

  etc.

* The label images have to have identical or similar names to the original pixel   
  images defined in *image_path*.

  This works well: Pixel and label images are located in different folders and have
  identical names:

   ```
   pixel_image_data/
   ├── leaves_1.tif
   ├── leaves_2.tif
   ├── leaves_3.tif
   └── leaves_4.tif

   label_image_data/
   ├── leaves_1.tif
   ├── leaves_2.tif
   ├── leaves_3.tif
   └── leaves_4.tif
   ```

  This works also: Pixel and label images are located in different folders and have
  similar names:

   ```
   pixel_image_data/
   ├── leaves_1.tif
   ├── leaves_2.tif
   ├── leaves_3.tif
   └── leaves_4.tif

   label_image_data/
   ├── leaves_1_labels.tif
   ├── leaves_2_labels.tif
   ├── leaves_3_labels.tif
   └── leaves_4_labels.tif
   ```  
Especially in case of multidimensional images:  Make sure to always
convert your label masks in *tif* format with [Fiji](https://fiji.sc) before using YAPiC.
Large amounts of image data can be conveniently converted with Fiji by using
[batch processing](https://imagej.net/Batch_Processing).   


## Optional parameters

### -n --normalize=NORM

Set pixel normalization scope [default: local]

* For minibatch-wise normalization choose ```local_z_score``` or ```local```.

* For global normalization use ```global_<min>+<max>``` (e.g. ```global_0+255``` for 8-bit images and ```global_0+65535``` for 16-bit images)

* Set to ```off``` to deactivate.

### --cpu

Train using the CPU (not recommended, very slow).

### --gpu=VISIBLE_DEVICES

If you want to use specific GPUs. To use gpu 0, set ```--gpu=0```. To use gpus 2 and 3, set ```--gpu=2,3```.

### -h --help               

Show documentation.

### --version

Show version.


## Train options

### -e --epochs=MAX_EPOCHS

Maximum number of epochs to train [default: 5000].

### -a --augment=AUGMENT

Set augmentation method for training [default: ```flip```].

* Choose ```flip``` and/or ```rotate``` and/or ```shear```.
* Use ```+``` to specify multiple augmentations (e.g. ```flip+rotate```).


###  -v --valfraction=VAL

Fraction of images to be used for validation [default: ```0.2```] between
0 and 1.


###  -f --file=CLASSIFER

Path to trained model [default: ```model.h5```].

### --steps=STEPS

Steps per epoch [default: ```50```].


### --equalize

Equalize label weights to promote influcence of less frequent labels.

###  --csvfile=LOSSDATA

Path to csv file for training loss data [default: ```loss.csv```].
