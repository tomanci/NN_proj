# Neural Network Project
*Made by*: **Ancilli Tommaso, University of Siena**

## How to use the script
To run the script you have to pick out one out of main_colab or main_local.

*main_colab* was created due to the necessity to have a GPU to make the experiments. The only requirement is to have a folder called NN_dataset and the zipped dataset in it. Once this requiriments has been fulfilled you can freely run the script.

*main_local* is 


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

------------------

This section will be delated before providing to him the scripts. This part contains the motiviation behind the choices taken in the construnction of the architecture, aming to not forget what i did and why

*Generator*: I try to replicate an autoencoder but on the contrary of the classical form - bottle-neck shape- I have done the opposite, where in the middle it is the wider. The width/breadth is given by the number of channles stuck together. Online the initial value provided to the net is flat ( a random vector) while I have decided to use directly a image, generated randomly, with the same dimention as the output. In the biggest conv layer I tried to implement a depthwise convolution, trying to reduce the computational burden that I have to perform. Here I have recreated same functions as those implemented in "Pre Processing". 

The activation function in the first two layers are LeakyReLu so that I want that the information flows as much as possible then I apply the classic ReLu


*Discriminator:* Its construnction was straightforward, meaning that i replicate, in my own way the theory found on the internet. Having to go from a 64x64x3 to a flat rapresentation I have to downscaled the image. In the first two layers I use a stride and a padding which allow me not to loose any short of information, indeed the output of the conv is like the input. To reduce the information, I have applied a AvgPooling with kernel 2x2 and stride 2. In the other layers the stride has been changed and this modification allows to halved the dimention of the image at each layer. As the information travels throught the net, the depth increases while the width/height shrinks. Once the data is 4x4x256, the we flat everything. 

*Pre Processig:* cropped the image -passing from 218x180 to 140x140, risize to a manageable size(64x64) converted to black-and-white. I have taken randomly slightly than 1/4 of the data and those will be used to generate faces.
Moreover, other functions have been added. Indeed one shuffle the dataset randomizing it, while the other two are needed to construct a mini batch sample to feed the discriminator. One of them has only to load the image and convert it into a tensor while the other has to append all the images into a 4D tensor. 


-------- DATA USED -------

Con la CNN + Autoencoder (Generator) non ho ottenuto risultati: ovvero dopo poce epoce, vi era una saturazione e ne il discrimator o generator eran in grado di imparare ed erano fermi. ( si vede dalla loss che non scende)

Quindi sono passato all'archittettura che su internet viene utilizzata.

primo problema: alla 3 epoca, circa con learning rate uguale per entrambi, il discriminator "sopprimeva" il generator, nel senco che il gioco di min-max volgeva in favore del discriminator. Entrambe le loss convergenvano verso 0, risultanto dell'impossibilità di apprendere da parte del generator.
Soluzione (a):
cambiato il learning rate, dopo varie prove lr_g = 1*10^-3; lr_d = 1*10^-5
Soluzione (b):
cambiato funzione di ottimmizazione. Invece di massimizzare sum(- (1-y)log(1-d(g))). cerco di minimizzare sum(-log(d(g))). 
