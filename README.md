# Neural Network Project
*Made by*: **Ancilli Tommaso, University of Siena**

## How to use the script
To run the script you have to pick out one out of main_colab or main_local. The only aspect that needs to be set-up before running the script is the presence of the dataset. I am refering to not the original one, called "celeba-dataset.zip" but the "img_celeba_dataset.zip" - coming from after manually cleaning the original dataset.


*main_colab* was created due to the necessity to have a GPU to make the experiments. The only requirement is to have a folder called NN_dataset and the zipped dataset in it. Once this requiriments has been fulfilled you can freely run the script. 

The only difference in terms of packages is the presence of matlablib to plot the loss function of both Generator and Discriminator for each mini-batch. 

*main_local* contains all the other scripts. To run the script you have to pass from inputs, throught the command line, the parameters. Again you have to pay attantion and locate exactly the position of your zipped file. The final dataset, composed of the subset of the original data taken for our purposes, will be placed on your current folder.


## Brief description of the files

**ANN.py** contains the classes of the neural networks implemented. 
In the *Generator* class, two different architecures were implemented. 

The first one can be selected if the perameter *architecture_Generator* = "CNN". This was my first attempt to construct a generative net setting out to replicate an autoencoder. Its structure is not the classical one -bottleneck shape- but in the middle is wider. In this attempt, the length and height of the image is fixed (64x64) and the breadth is changing, going from 3 up to 256 and then back at 3. This architecture shows important limits and it pushes me to develop the other one *architecture_Generator* = "conventional".


This time the architecture resambles the one which can be found in the lecterature. It takes as input a small tensor (4,4,128) and layer after layer we decrease simultaneously the depth while augmenting the resolution going from 4x4 up to 64x64. This architecture is a game-changer, indeed it was able to withstand the Discrimintator during the min-max game. Despite deploying two different learning rate, the previous Generator cannot cope with the Discriminator which outperform the Generator in detecting real from generated images. 



------------------

**Dataset_creator.py** contains the method to download and pre-processing the dataset. 

------------------

**Requirements.py** loads the libraries.

------------------

**main_local.py** This file contains all the call to the different scripts. If you want to run the code locally you should use this file. From terminal, there is the possibility to manually select all the parameters

------------------

**main_colab.ipynb** I created this file because I have to run the code on Colab to speed-up the computations. This is self contained, meaning that it has in each cell the code of the previous scripts. 

------------------

**Train_valuation.py** contantains the traing functions. You will find two different functions you can choose from. 
The fist one called as *train_lr* implements the first version of the training phase where the Generator aims to maximize the Binary Cross-Entropy while the Discriminator tries to achieve the opposite. 
The last one defined as *train_lr_obj* deploys a slightly modified version of the function. Indeed the Generator does minimize the following function $E[log(D(G(z))]$ as it was suggested by the original paper.  

