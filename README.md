# SingleClassRL


Implementation of weak single object segmentation from paper "Regularized Loss for Weakly Supervised Single Class Semantic Segmentation", ECCV2020.
[PDF](https://cs.uwaterloo.ca/~oveksler/Papers/eccv2020.pdf) 

## Main Files
* train_with_anneal.py: use for training first in annealing stage, then in normal stage
* train_with_transfer.py: use from training with weight transfer from another dataset, models that can be used for weight transfer are in directory 'trained_models'

## OxfodPet dataset
Download OxfordPet from  (https://www.robots.ox.ac.uk/~vgg/data/pets/)
Files in 'SingleClassRL\data\Oxford_iit_pet\annotations' should be placed in the 'annotation' directory of OxfordPet dataset
