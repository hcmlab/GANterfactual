# GANterfactual
Generating Counterfactual Explanation Images through Generative Adversarial Learning. 

![Example Counterfactual Images](https://github.com/hcmlab/GANterfactual/blob/main/ressources/counterfactual_examples.PNG)

This repository aims to provide the code used in the paper "This is not the Texture you are looking for! Introducing Novel Counterfactual Explanations for Non-Experts using Generative Adversarial Learning". A preprint of the paper can be found [here](https://arxiv.org/abs/2012.11905).

The dataset used in the paper can be downloaded [here](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).
Once you have downloaded it, you can split it into the required data partitions by using the *preprocessor.py* script that can be found in the *data* directory.
Run it by executing the following command in a terminal:

`python preprocessor.py -i path_of_raw_dataset -t 20 -v 10 -d 512`

This will split your dataset into *train*, *validation* and *test* partitions (70% train, 20% test, 10% validation). The data will be written to the respective subdirectories in the *data* directory.

To train the classifier that was used in the paper, use the *train_alexNet.py* script in the *GANterfactual* directory:

`python train_alexNet.py`

The final classifier model will be stored at *models/classifier/model.h5*

To train the the GANterfactual model, that consists of a CycleGAN that is modified with an additional counterfactual loss component, use the *cyclegan.py* script in the *GANterfactual* directory:

`python cyclegan.py`

The four parts of the final GANterfactual model will be stored at *models/GANterfactual*.

Our used inference scripts will be added soon.

Our pre-trained models will be provided for download soon as well.
