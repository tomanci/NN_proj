# Neural Network Project
*Made by*: **Ancilli Tommaso, University of Siena**

## How to use the script

## Brief description of the files
**ANN.py** contains the classes of the neural networks implemented


**Dataset_creator.py** contains the method to download and pre-processing the dataset

**Requirements.py** loads the libraries

------------------

This section will be delated before providing to him the scripts. This part contains the motiviation behind the choices taken in the construnction of the architecture, aming to not forget what i did and why

*Generator*: I try to replicate an autoencoder but on the contrary of the classical form - bottle-neck shape- I have done the opposite, where in the middle it is the wider. The width/breadth is given by the number of channles stuck together. Online the initial value provided to the net is flat ( a random vector) while I have decided to use directly a image, generated randomly, with the same dimention as the output. In the biggest conv layer I tried to implement a depthwise convolution, trying to reduce the computational burden that I have to perform. Here I have recreated same functions as those implemented in "Pre Processing". 

The activation function in the first two layers are LeakyReLu so that I want that the information flows as much as possible then I apply the classic ReLu


*Discriminator:* Its construnction was straightforward, meaning that i replicate, in my own way the theory found on the internet. Having to go from a 64x64x3 to a flat rapresentation I have to downscaled the image. In the first two layers I use a stride and a padding which allow me not to loose any short of information, indeed the output of the conv is like the input. To reduce the information, I have applied a AvgPooling with kernel 2x2 and stride 2. In the other layers the stride has been changed and this modification allows to halved the dimention of the image at each layer. As the information travels throught the net, the depth increases while the width/height shrinks. Once the data is 4x4x256, the we flat everything. 

*Pre Processig:* cropped the image -passing from 218x180 to 140x140, risize to a manageable size(64x64) converted to black-and-white. I have taken randomly slightly than 1/4 of the data and those will be used to generate faces.
Moreover, other functions have been added. Indeed one shuffle the dataset randomizing it, while the other two are needed to construct a mini batch sample to feed the discriminator. One of them has only to load the image and convert it into a tensor while the other has to append all the images into a 4D tensor. 