class Generator(): #può sia essere il Generator nelle GAN che una classica NN negli adversarial attack
    TODO #1
    TODO #2
    TODO #3
    """
    input: tensor (64,64,3) x,y,z dimension

    output: tenror (64,64,3) xyz

    Inner structure: an autoencoder where the third layer is the deepest 
    """

    def __init__(self):

        self.layers = []

        self.layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1)) #stride = 1 default
        self.layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))#layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Conv2d(64, 128, kernel_size=5,padding=2)) #stride = 1 default
        self.layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        #depthwise separable CNN layer
        self.layers.append(nn.Conv2d(128,128,kernel_size=7,padding=3,groups=128))
        self.layers.append(nn.Conv2d(128,256,kernel_size=1))
        #self.layers.append(nn.Conv2d(128, 256, kernel_size=7, padding=3)) #stride = 1 default
        self.layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.layers.append(nn.Conv2d(256, 64, kernel_size=3, padding=1)) #stride = 1 default
        self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Conv2d(64, 3, kernel_size=3, padding=1)) #stride = 1 default
        self.layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*self.layers)

    
    def forward(self, x_input):

      self.output = self.net(x_input)*255

      return self.output


    def image_generation(self):
      """
      This function will be used to display a image or more after ending each batch/epoch...
      """

      random_input = torch.rand((3,64,64))
      transform = T.ToPILImage()
      im =transform(self.forward(random_input))

      return im.show()

    def summary(self):
    
      total_param = 0
      param_string = ""

      for l in self.layers:
        parameters = 0

        if "Conv" in str(l):
          param_string = param_string + "======================================================" + "\n"
          param_string = param_string + str(l) + "\n"
          param_string = l.in_channels * l.kernel_size[0]*l.kernel_size[1]*l.out_channels + l.out_channels
          param_string = param_string + "Number of parameters for this layer: {}".format(parameters) + "\n"

          total_param = total_param + parameters
        
        else:
          param_string = param_string + "======================================================" + "\n"
          param_string = param_string + str(l) + "\n"
          param_string = param_string + "Number of parameters for this layer:{} ".format(parameters)+"\n"

      param_string = param_string + "----------------------------------------------------" + "\n"  
      param_string = param_string + "Total number of learnable parameters: {}".format(total_param)

    
      print(param_string)



class Discriminator():
    def __init__(self):
        self.layers = []

        self.layers.append(nn.Conv2d(3,32,kernel_size=7,padding = "same"))#64x64x32
        self.layers.append(nn.AvgPool2d(2,stride=2))#32x32x32
        self.layers.append(nn.LeakyReLU(inplace=True))

        self.layers.append(nn.Conv2d(32,64,kernel_size=5,padding="same"))#32x32x64
        self.layers.append(nn.AvgPool2d(2,stride=2))#16x16x64
        self.layers.append(nn.LeakyReLU(inplace=True))

        self.layers.append(nn.Conv2d(64,128,kernel_size=3,padding=1,stride=2))#8x8x128
        self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Conv2d(128,256,kernel_size=3,padding=1,stride=2))#4x4x256
        self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Flatten())#256X16

        self.layers.append(nn.Linear(in_features=256*4*4,out_features=1))

        self.layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.layers)

        #MANCANDO DA METTERCI LE ReLU oppure BatchNormalization o cose simili 