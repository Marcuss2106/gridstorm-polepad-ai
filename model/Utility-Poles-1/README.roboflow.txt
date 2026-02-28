
Utility Poles - v1 2026-02-28 3:15pm
==============================

This dataset was exported via roboflow.com on February 28, 2026 at 8:21 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 912 images.
Telephone-poles are annotated in YOLO26 format.

The following pre-processing was applied to each image:
* Resize to 512x512 (Stretch)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 3 versions of each source image:
* Randomly crop between 0 and 15 percent of the image
* Random rotation of between -10 and +10 degrees
* Random shear of between -5° to +5° horizontally and -5° to +5° vertically
* Random exposure adjustment of between -10 and +10 percent
* Random Gaussian blur of between 0 and 1 pixels


