# Semantic Segmentation
Udacity nanodegree Advanced Deep Learning project

## Development System
Nvidia GTX 1080 with 8GB on a quad core i7 with 32GB running Linux Mint.

## Architecture
This project uses Fully Convolutional Network constructed from a vgg16 pretrained model followed by 1x1 convolution filters.  The 1x1 convolutions connect layers 3, 4, and 7 to transposed convolution filters that upsample the classifier outputs to enable pixel-level prediction, in this case whether a pixel is or is not road.  While the specific settings are tuned for this project, the method follows that described by Long, Shelhamer, and Darrell in *Fully Convolutional Networks for Semantic Segmentation.*

## Examples of Segmentation
As shown in the following images, where green marks the predicted road surface, the trained network's inference can be quite good.  However, as shown in the following section, performance is not always this good.

###Urban Marked

![um_000056 latest](runs/latest/um_000056.png "Latest result um_000056")

![um_000025 latest](runs/latest/um_000025.png "Latest result um_000025")

###Urban Multiple Marked

![umm_000017 latest](runs/latest/umm_000017.png "Latest result umm_000017")

![umm_000031 latest](runs/latest/umm_000031.png "Latest result umm_000031")

###Urban Unmarked

![uu_000071 latest](runs/latest/uu_000071.png "Latest result uu_000071")

![uu_000084 latest](runs/latest/uu_000084.png "Latest result uu_000084")

## Observations
Running the same configuration twice can produce considerably different results in some cases.  The image below is um_000092 from the latest run.  Following is the same image from the previous run.  In both cases, the networks were trained for 250 epochs on an augmented data set of 1156 images.  In some cases a shorter training session produces better results on some images than a longer run with the same configuration.  *Note that the longer run is not a continuation of the shorter run.*

Latest network, 250 epochs

![um_000092 latest](runs/latest/um_000092.png "Latest result")

Prior network, 250 epochs

![um_000092 prior](runs/prior/um_000092.png "Previous result")

### Epochs do not always improve prediction
250 epochs

![um_000080 latest](runs/latest/um_000080.png "Latest result")

100 epochs

![um_000080 prior](runs/prior/um_000080.png "100 epoch result")

