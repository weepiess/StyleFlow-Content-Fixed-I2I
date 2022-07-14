# Code for StyleFlow
## a. Requirements:
* Python 3.6
* Numpy 1.17.3
* Pytorch 1.5.0
* torchvision 0.6.0
* tensorboard 2.0.0

## b. How to Train
### 1. MS-COCO to WikiArt
* Prepare the dataset
* Prepare the source & target path
~~~ bash
#example
/path/COCO_train2014_000000188902.jpg
/path/COCO_train2014_000000139276.jpg
~~~
* Modify the root path in **wikiart.yaml**
~~~ yaml
rootA: /path/src/train.txt

rootB: /path/target/train.txt
~~~
* run scripts **train_wikiart.sh**

### 2. GTA to Cityscapes
* Prepare the dataset
* Prepare the source & target path
~~~ bash
#example
/path/xxx.jpg
/path/xxx.jpg
~~~
* Modify the root path in **gta.yaml**
~~~ yaml
rootA: /path/src/train.txt

rootB: /path/target/train.txt
~~~
* run scripts **train_gta.sh**