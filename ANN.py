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