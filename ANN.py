import torch.nn as nn
import torch
from torchvision import transforms 
import torchvision.transforms as T
 
class Generator(): 
    """
    #TODO #1
    #TODO #2
    #TODO #3
    #TODO #5
    #TODO #6
    """
    """
    input: tensor (5,5,5) x,y,z dimension

    output: tenror (64,64,3) xyz

    Inner structure: with the parameter "structure" an autoencoder net or a paper-based net can be deployed   
    """

    def __init__(self,processing_unit,structure):
        
      if structure == "CNN":
        self.layers = []

        self.layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1)) #stride = 1 default
        self.layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))#layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Conv2d(64, 128, kernel_size=5,padding=2)) #stride = 1 default
        self.layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        #depthwise separable CNN layer
        #self.layers.append(nn.Conv2d(128,128,kernel_size=7,padding=3,groups=128))
        #self.layers.append(nn.Conv2d(128,256,kernel_size=1))
        self.layers.append(nn.Conv2d(128, 256, kernel_size=7, padding=3)) #stride = 1 default
        self.layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.layers.append(nn.Conv2d(256, 64, kernel_size=3, padding=1)) #stride = 1 default
        self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Conv2d(64, 3, kernel_size=3, padding=1)) #stride = 1 default
        self.layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*self.layers)

        self.input_shape = (3,64,64)

        
      elif structure == "conventional":
        
        self.layers = []

        self.layers.append(nn.ConvTranspose2d(in_channels=5,out_channels=128,kernel_size=4,padding=2,stride=1))#[5,5,5] -> [4,4,128]
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(128))

        self.layers.append(nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,padding=2,stride=3))#[4,4,128] -> [8,8,64]
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(64))

        self.layers.append(nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,padding=1,stride=2))#[8,8,64] -> [16,16,32]
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(32))

        self.layers.append(nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=4,padding=1,stride=2))#[16,16,32] -> [32,32,16]
        self.layers.append(nn.ReLU())

        self.layers.append(nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=2,padding=0,stride=2))#[32,32,16] -> [64,64,3]
        self.layers.append(nn.ReLU())

        self.net = nn.Sequential(*self.layers)

        self.input_shape = (5,5,5)
        
      self.device = torch.device(processing_unit)
      self.net = self.net.to(self.device)
      self.position = 0


    
    def forward_G(self, x_input:torch.tensor):

      """
      ----- Input -----
      x_input -> tensor, dimension #example_mini_batch x #channels x width x height 

      ----- Output ----
      self.image_output -> tensor, dimension #example_mini_batch #channels x width x height 
      self.label_fake_img -> tensor, dimension 1 x #example_mini_batch

      """

      self.image_output = self.net(x_input)
      
      self.label_fake_img = torch.tensor([[0]*len(x_input)])

      return self.image_output,self.label_fake_img


    def function_loss_G(self,output_label_gen:torch.tensor,true_label:torch.tensor):
      """
      ----- INPUT -----
      output_label_gen -> tensor 1 x #batch_sample
      true_label -> tensor 1 x #batch_sample
      ----- OUTPUT -----

      loss function. (scalar value)

      """
      cost_function = nn.BCELoss()
      true_label = true_label.to(torch.float)
      loss = cost_function(output_label_gen,true_label)

      return loss


    def image_generation(self):
      """
      This function will be used to display a image after the ending of batch/epoch...
      """
      #creation of a tuple (1,self.input_shape)
      random_input = torch.rand(size= (1,)+self.input_shape)
      random_input = random_input.to(self.device)
      transform = T.ToPILImage()#function to transform a tensor into a image
      out = self.net(random_input)
      out = out.view(out.shape[0]*out.shape[1],out.shape[2],out.shape[3])
      im = transform(out)

      return im.show()


    def creation_image(self, length_dataset:int):
      """
      ----- INPUT -----
      length_dataset -> int value

      ----- OUTPUT ------
      a tensor image (img) and a boolean function if the we have reached the end of the dataset
      """

      img = torch.rand(self.input_shape)
      end_dataset = False
      
      if self.position <  length_dataset -1:
        self.position = self.position +1 
      else:
        self.position = 0
        end_dataset = True

      return img, end_dataset
    
    def input_creation(self,length_dataset:int,batch_size:int,current_batch_dim = None):
      """
      This function create a tensor for feeding the generator net. 
      If current_batch_dim = None, the output is feed into the discriminator without propagating the gradient till the generator. 

      ----- INPUT -----
      length_dataset -> int value 
      batch_size -> int value

      ----- OUTPUT -----
      batch_input, 4D tensor: #sample_batch x #channels x width x height  
      """
      if current_batch_dim == None:

        length_dataset = length_dataset
        i = 0
        self.batch_input = []

        last_batch = False

        while i < batch_size:

          img,end_dataset_flag = self.creation_image(length_dataset)
          self.batch_input.append(img)
          last_batch = end_dataset_flag

          i = i + 1
        
          if last_batch:
            break

        self.batch_input = torch.stack(self.batch_input,dim = 0)

      else:
        
        last_batch = False
        
        self.batch_input = torch.rand((current_batch_dim))


      return self.batch_input, last_batch


    def summary(self):
      """
      summary of the model 
      """
      for idx,l in enumerate(self.layers):
        print(idx+1, "->",l)
        if "LU" in str(l):
          print("\n")

    def save(self,file_name):

      torch.save(self.net.state_dict(), file_name)



class Discriminator():

    def __init__(self,processing_unit):
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

        self.device = torch.device(processing_unit)
        self.net = self.net.to(self.device)

    
    def combined_True_Fake(self, fake_labels:torch.tensor, fake_images:torch.tensor, true_labels:torch.tensor, true_images:torch.tensor):
      """
      The true and generated images are combined together 
        ----- Input ----
        fake_labels, fake_images, true_labels, true_images -> tensor
        dimensions:
        fake_images, true images = #of_example_mini_batch x #channles x width x height
        fake/true labels = 1x#of_example_mini_batch

        ----- Output ------
        combined_images, combined_label -> tensor
        dimension:
        combined_label = 1 x #examples_mini_batch
        combined_images = 2*#examples_mini_batch x width x height 

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

      return combined_images,combined_labels

    def forward_D(self,images:torch.tensor):
      """
      ----- INPUT -----
      image -> tensor 2*#batch_size x #channels x width x height

      ----- OUTPUT -----
      tensor -> 2*#batch_size x 1
      """
      self.output = self.net(images)
      self.output = self.output.to(torch.float32)
      return self.output 
      
    def function_loss_D(self,output_label_dis:torch.tensor,true_label:torch.tensor):
      """
      ----- INPUT -----
      output_label_dis -> 2*#batch_size x 1
      true_label -> 2*#batch_size x 1 
      
      ----- OUTPUT -----
      loss function (scalar value)

      """
      cost_function = nn.BCELoss()
      true_label = true_label.to(torch.float)
      loss = cost_function(output_label_dis,true_label)

      return loss

    def save(self,file_name):
      torch.save(self.net.state_dict(), file_name)

    
    def summary(self):
      """
      summary of the model 
      """
      for idx,l in enumerate(self.layers):
        print(idx+1, "->",l)
        if "LU" in str(l):
          print("\n")
