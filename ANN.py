class Generator(): #può sia essere il Generator nelle GAN che una classica NN negli adversarial attack
    TODO #1
    TODO #2
    TODO #3
    TODO #5
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

        self.position = 0

    
    def forward(self, x_input):

      """
      ----- Input -----
      x_input -> tensor, dimension #example_mini_batch x #channels x width x height 

      ----- Output ----
      self.image_output -> tensor, dimension #example_mini_batch #channels x width x height 
      self.label_fake_img -> tensor, dimension 1 x #example_mini_batch

      """

      self.image_output = self.net(x_input)*255
      
      self.label_fake_img = torch.tensor([[0]*len(x_input)])

      return self.image_output,self.label_fake_img


    def image_generation(self):
      """
      This function will be used to display a image or more after ending each batch/epoch...
      """

      random_input = torch.rand((3,64,64))
      transform = T.ToPILImage()#function to transform a tensor into a image
      im = transform(self.forward(random_input))

      return im.show()


    def creation_image(self, length_dataset):
      """
      img tensore creato
      copntrolla lunghezza dataset e si accorge di essere dentro; se si mi stampa l'immagine 
      """

      img = torch.rand((3,64,64))
      end_dataset = False
      
      if self.position <  length_dataset -1:
        self.position = self.position +1 
      else:
        self.position = 0
        end_dataset = True

      return img, end_dataset
    
    def input_creation(self,length_dataset,batch_size):
      """
      concatena più immagini a formare un tensore 4D. I label non ne ho bisofno in quanto li creo in forward che so di già come sono, ovvero tutti falsi 
      
      """
      length_dataset = length_dataset
      i = 0
      self.batch_input = []

      last_batch = False

      while i < batch_size:

        img,end_dataset_flag = self.creation_image(length_dataset)
        self.batch_input.append(img)
        last_batch = end_dataset_flag

        i = i + 1
      
        if not last_batch:
          break

      batch_input = torch.stack(batch_input,dim = 0)

      return batch_input 

    """
    DA TESTARE LE DUE FUNZIONI... SONO COME QUELLE IN DATASET QUINDI DOVREBBE ESSRE ABBASTANZA VELOCE COME COSA
    """

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

    
    def shuffled_True_Fake(self, fake_labels, fake_images, true_labels, true_images):
      """
        ----- Input ----
        fake_labels, fake_images, true_labels, true_images -> tensor
        dimensions:
        fake_images, true images = #of_example_mini_batch x #channles x width x height
        fake/true labels = 1x#of_example_mini_batch

        ----- Output ------
        shuffled_images, shuffled_label -> tensor
        dimension:
        shuffled_label = 1 x #examples_mini_batch
        shuffled_images = 2*#examples_mini_batch x width x height 

      """

      if len(fake_labels) != len(true_labels) and fake_images.shape != true_images.shape:
        raise ValueError("THE DIMENSIONS ARE NOT THE SAME!")

      combined_images = torch.stack([true_images,fake_images],dim=0)
      # combined_images dimension = 2(number of tensor combined into a list) x #mini_batches x channels x width x height
      combined_images = combined_images.view(
        combined_images.shape[0]*combined_images.shape[1],
        combined_images.shape[2],combined_images.shape[3],combined_images.shape[4]
      )
      #with view I reshape the dimension, compressing the first two dimention into a single one.
      #it's like pilling up the tensor one after the other in the dimension of the mini batches 

      #hstack -> it stacks the tensor horizionally, so from two tensor of shape [1,n] and [1,n] it prints out [1,2*n]
      combined_labels = torch.hstack([true_labels,fake_labels])
      combined_labels = combined_labels.view(-1,1)
      #view(-1,1) it's like numpy.reshape and it "transpose" creating a [2*n,1] vector 

      shuffled_vector = torch.randperm(combined_images.shape[0])

      shuffled_images = [combined_images[i] for i in shuffled_vector]
      shuffled_labels = [combined_labels[i] for i in shuffled_vector]

      shuffled_images = torch.stack(shuffled_images,dim=0)
      shuffled_labels = torch.stack(shuffled_labels,dim =0)

      return shuffled_images,shuffled_labels
      
