class Dataset():

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
            
        
    def preprocessing(self):
        import os
        import PIL
        from PIL import Image

        #creation of the path where the initial dataset is present
        #self.path_destination = #current_path = os.path.join(os.getcwd(),"img_celeba_dataset")

        #creation of the path where the new dataset will be stored
        path_dataset = os.path.join(os.getcwd(),"dataset")
        os.mkdir(path_dataset)

        for photo in os.listdir(self.path_destination):
            #directory of the photos
            dir = os.path.join(self.path_destination,photo)

            #in sequence, I open the photo, crop it, reshape and convert into a greyscale 
            im = Image.open(dir).crop((30,55,150,175)).resize((64,64)).convert("L")
            #save the new image into that directory
            im.save(str(path_dataset+"/"+photo))





