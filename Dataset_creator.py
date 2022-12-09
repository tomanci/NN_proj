class Dataset():

    def __init__(self,path,device):
        self.path = path 
        self.device = device

        if self.device == "colab":
            import os
            !pip install unzip
            from google.colab import drive
            drive.mount('/content/gdrive')
            ! pip install unzip
            !cp /content/gdrive/MyDrive/NN_dataset/img_celeba_dataset.zip /content
            !unzip "/content/img_celeba_dataset.zip" -d "/content"
        
        elif self.device == "local":
            


