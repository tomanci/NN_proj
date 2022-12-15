class Dataset():
    """
    We transfer the kaggle dataset into a folder where the python environment is. 
    path_source is where the kaggle dataset downloaded is, path destination is where it should be placed the unzipped dataset.[by default there are my paths] 
    """

    def __init__(self,device,path_source = "/Users/tommasoancilli/Downloads/img_celeba_dataset.zip",path_destination= "/Users/tommasoancilli/Desktop/Python/NN_proj/img_celeba_dataset"):

        self.path_source = path_source
        self.path_destination = path_destination
        self.device = device

        if self.device == "colab":
            !pip install unzip
            from google.colab import drive
            drive.mount('/content/gdrive')
            !cp /content/gdrive/MyDrive/NN_dataset/img_celeba_dataset.zip /content
            !unzip "/content/img_celeba_dataset.zip" -d "/content"
        
        elif self.device == "local":

            import shutil

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

        import os
        import PIL
        from PIL import Image

        self.p_subset = p_subset

        #creation of the path where the initial dataset is present
        #self.path_destination = #current_path = os.path.join(os.getcwd(),"img_celeba_dataset")

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

                #in sequence, I open the photo, crop it, resize(downsampled to a 64x64 resolution) and convert into a greyscale  
                im = Image.open(dir).crop((20,45,150,185)).resize((64,64)).convert("L")#140x140 before resize #crop((30,55,150,175))#120x120
                #save the new image into that directory
                im.save(str(self.path_dataset+"/"+"real_"+photo))

        print("counter photos {} counter dataset {} ratio {}".format(self.counter_photos,self.counter_dataset, self.counter_dataset/self.counter_photos))

    
    def shuffled_dataset(self):

        """
        mixed the dataset, by returning a list containing the path of every single image, which will be needed when we have to upload the mini batch
        a stupid example of an entries of self.files is like: "/Users/tommasoancilli/Desktop/Python/NN_proj/dataset/real_8328239.jpg"
        """

        self.files = [os.path.join(self.path_dataset,f) for f in os.listdir(self.path_dataset)]
        self.true_labels = [1]*len(self.files)

        shuffled_list = torch.randperm(len(self.files))

        self.files = [self.files[i] for i in shuffled_list]
        self.true_labels = [self.true_labels[i] for i in shuffled_list]

        return self.files, self.true_labels








