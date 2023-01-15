import shutil 
import PIL
import os 
from PIL import Image
import torch
from torchvision import transforms 
import torchvision.transforms as T

class Dataset():
    """
    We transfer the kaggle dataset into a folder where the python environment is. 
    path_source is where the kaggle dataset downloaded is, path destination is where it should be placed the unzipped dataset.[by default there are my paths] 
    """
    """#TODO #4"""
    def __init__(self,path_source = "/Users/tommasoancilli/Downloads/img_celeba_dataset.zip",path_destination= "/Users/tommasoancilli/Desktop/Python/NN_proj/img_celeba_dataset"):

        self.path_source = path_source
        self.path_destination = path_destination
        self.file_id = 0
        self.files = []
        self.true_labels = []

        shutil.unpack_archive(self.path_source, self.path_destination)    
            
        
    def preprocessing(self,p_subset=0.25):
        
        """
        preprocessing function creates the dataset folder where a subset of the original images is taken 
        --- INPUT ---
        p_subset = the percentage of the image transfered to the dataset default = 0.25

        --- OUTPUT ---
        dataset folder with the preprocessed images. 
        Each image has undergone to the following processes
        1) crop it and centralized 
        2) downsampled to 64x64 resolution
        3) converted from a RGB to a grayscale image [computational reason]
        """

        self.p_subset = p_subset

        #creation of the path where the initial dataset is present
        self.path_destination = str(self.path_destination+"/img_celeba_dataset")

        #creation of the path where the new dataset will be stored
        self.path_dataset = os.path.join(os.getcwd(),"dataset")
        os.mkdir(self.path_dataset)

        self.counter_photos = 0
        self.counter_dataset = 0

        for photo in os.listdir(self.path_destination):

            self.counter_photos = self.counter_photos +1
            if torch.rand(1) <= p_subset:

                self.counter_dataset = self.counter_dataset +1
                #directory of each photo example (../img_celeba_dataset/19736.jpeg)
                dir = os.path.join(self.path_destination,photo)

                #in sequence, I open the photo, crop it, resize(downsampled to a 64x64 resolution) 
                im = Image.open(dir).crop((20,45,150,185)).resize((64,64))#140x140 
                #save the new image into that directory
                im.save(str(self.path_dataset+"/"+"real_"+photo))

        print("counter photos {} counter dataset {} ratio {}".format(self.counter_photos,self.counter_dataset, self.counter_dataset/self.counter_photos))

    
    def shuffled_dataset(self,boolean_shuffle=True):

        """
        mixed the dataset, by returning a list containing the path of every single image, which will be needed when we have to upload the mini batch
        a trivial example of an entries of self.files is like: "/Users/tommasoancilli/Desktop/Python/NN_proj/dataset/real_8328239.jpg"
        """
        self.path_dataset = os.path.join(os.getcwd(),"dataset/")

        self.files = [os.path.join(self.path_dataset,f) for f in os.listdir(self.path_dataset)]
        self.true_labels = [1]*len(self.files)

        if boolean_shuffle:
            shuffled_list = torch.randperm(len(self.files))

            self.files = [self.files[i] for i in shuffled_list]
            self.true_labels = [self.true_labels[i] for i in shuffled_list]

        return self.files, self.true_labels


    def load_and_conversion(self,file_img):

        """
        it loads and converts the images into a tensor format 
        ----- INPUT -----

        ----- OUTPUT -----

        img_tensor : tensor of the image, being converted from a .jpg format to a tensor one
        label_tensor : tensor of the label 
        end_dataset : flag showing the bottom of the dataset 
        """
        self.files = file_img
        self.end_dataset = False

        im = Image.open(self.files[self.file_id])
        label = self.true_labels[self.file_id]
         
        convert_tensor = transforms.ToTensor()

        img_tensor = convert_tensor(im)
        label_tensor = torch.tensor(label)

        if self.file_id < len(self.files)-1:
            self.file_id = self.file_id +1
        else:
            self.end_dataset = True
            self.file_id = 0 #reset the index


        return img_tensor,label_tensor,self.end_dataset



    def mini_batch_creation(self, batch_size:int,file_img):
        """
        create the a batch containing the images processed
        ------ INPUT ------
        batch_size: given by the users 

        ----- OUTPUT -----
        mini_batch_image: batch of images, its shape is a 4D tensor #batch_size x #channels x width x height
        mini_batch_labels: batch of labels, its a tensor: 1 x #batch_size 
        last_batch_flag: flag variable to say when we have reached the last batch. In this way, in the training function it will indicate the end of a epoch 


        """
        i = 0
        self.data_batch_im = []
        self.data_batch_label = []

        last_batch_flag = False

        while i < batch_size:
            
            img,label,end_dataset_flag = self.load_and_conversion(file_img)

            self.data_batch_im.append(img)
            self.data_batch_label.append(label)
            last_batch = end_dataset_flag

            i = i + 1

            if last_batch:
                last_batch_flag = True
                break
        
        mini_batch_image = torch.stack(self.data_batch_im,dim=0)
        mini_batch_label = torch.stack(self.data_batch_label,dim=0).view(1,-1)

        return mini_batch_image, mini_batch_label, last_batch_flag



