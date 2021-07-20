[![Build Status](https://travis-ci.com/yapic/yapic.svg?branch=master)](https://travis-ci.com/yapic/yapic)
[![PyPI](https://img.shields.io/pypi/v/yapic.svg?color=green)](https://pypi.org/project/yapic/)

<!-- full url is needed for images to be rendered on pypi -->
![DZNE](https://raw.githubusercontent.com/yapic/yapic/master/docs/img/DZNE_CMYK_E.png)<!-- .element height="50%" width="50%" -->

YAPiC is developed by
[Image and Data Analysis Facility](https://www.dzne.de/forschung/core-facilities/image-and-data-analysisfacility/), [Core Reseach Facilities](https://www.dzne.de/forschung/core-facilities/), [DZNE](https://www.dzne.de/en) (German Center for Neurodegenerative Diseases).


# YAPiC - Yet Another Pixel Classifier (based on deep learning)

Check the [YAPiC Website](https://yapic.github.io/yapic/) for documentation,
examples and installation instructions.


## What is YAPiC for?

With YAPiC you can make your own customzied filter (we call it *model* or *classifier*) to enhance a certain structure of your choice.

We can, e.g train a model for detection of oak leafs in color images, and use this oak leaf model to filter out all image regions that are not covered by oak leaves:

![](https://raw.githubusercontent.com/yapic/yapic/master/docs/img/oak_example.png "oak leaf classifier example")

* Pixels that belong to other leaf types
  or to no leafs at all are mostly suppressed, they appear dark in the output image.
* Pixels that belong to oak leafs are enhanced, they appear bright in the output image.

The output image is also called a *probability map*, because the intensity of each pixel corresponds to the probability of the pixel belonging to an oak leave region.

You can train a model for almost any structure you are interested in, for example to detect a certain cell type in histological micrographs (here: purkinje cells of the human brain):

![](https://raw.githubusercontent.com/yapic/yapic/master/docs/img/histo_example.png "purkinje cell classifier example")
*Histology data provided by Oliver Kaut (University Clinic Bonn, Dept. of Neurology)*

We have used YAPiC for analyzing various microscopy image data. Our experiments are mainly related to neurobiology, cell biology, histopathology  and drug discovery (high content screening).
However, YAPiC is a very generally applicable tool and can be applied to very different domains. It could be used for detecting e.g. forest regions in satellite images, clouds in landscape photographs or fried eggs in food photography.
