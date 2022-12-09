class Generator: #può sia essere il Generator nelle GAN che una classica NN negli adversarial attack

    """
    input: tensor (64,64,3) x,y,z dimension

    output: tenror (64,64,3) xyz

    Inner structure: an autoencoder where the third layer is the deepest 
    """

    def __init__(self):

        layers = []

        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1)) #stride = 1 default
        layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))#layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, 128, kernel_size=5,padding=2)) #stride = 1 default
        layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        layers.append(nn.Conv2d(128, 256, kernel_size=7, padding=3)) #stride = 1 default
        layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        layers.append(nn.Conv2d(256, 64, kernel_size=3, padding=1)) #stride = 1 default
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, 3, kernel_size=3, padding=1)) #stride = 1 default
        layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    
    def forward(self, x_input):

      self.output = self.net(x_input)*256

      return self.output