from Requirements import *
from ANN import *
from Dataset_creator import *
from Train_valuation import *

def parse_command_line_arguments():
    """Parse command line arguments, checking their values."""

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('architecture_Generator', type = str, choices=['CNN', 'conventional'],
                        default = "conventional",help=' you can choose between a CNN or Conventional architecture')
    parser.add_argument('lr_g', type=float, default=0.001,
                        help='learning rate (Adam) (default: 0.001) for GENERATOR')
    parser.add_argument('lr_d', type=float, default=0.00001,
                        help='learning rate (Adam) (default: 0.00001) for DISCRIMINATOR')
    parser.add_argument('path_source', type=str, default = "/Users/tommasoancilli/Downloads/img_celeba_dataset.zip",
                        help='where the downloaded dataset is stored: mine was in /Users/tommasoancilli/Downloads/img_celeba_dataset.zip')
    parser.add_argument('path_destination', type=str, default = "/Users/tommasoancilli/Desktop/Python/NN_proj/img_celeba_dataset",
                        help='where you want to store the unzipped dataset: mine was in /Users/tommasoancilli/Desktop/Python/NN_proj/img_celeba_dataset')
    parser.add_argument('p_subset', type=float, default=0.25,
                        help='fraction of data kept for generating the images')
    parser.add_argument('batch_size', type=int, default=128,
                        help='mini-batch size (default: 128)')
    parser.add_argument('epochs', type=int, default=15,
                        help='number of training epochs (default: 15)')
    parser.add_argument('training', type=str, default='lr', choices=['lr', 'lr_obj'],
                        help='which type of training function you want to use (default: train_lr)"')

    parsed_arguments = parser.parse_args()

    return parsed_arguments


if __name__ == "__main__":

    args = parse_command_line_arguments()

    processing_unit = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for k,v in args.__dict__.items():
        print(k + '=' + str(v))
    if args.architecture_Generator == "conventional" or args.architecture_Generator == "CNN":
        gen_net = Generator(processing_unit=processing_unit,structure = args.architecture_Generator)
    else:
        raise ValueError("Invalid architecture type!")

    dis_net = Discriminator(processing_unit=processing_unit)

    dataset = Dataset(path_source=args.path_source,path_destination=args.path_destination)
    if args.training == "lr":
        train_lr(architecture_G=gen_net,architecture_D=dis_net,batch_size=args.batch_size,
                n_epochs=args.epochs,dataset=dataset, lr_g=args.lr_g, lr_d=args.lr_d, 
                processing_unit=processing_unit,p_subset=args.p_subset)

    elif args.training == "lr_obj":
            train_lr_obj(architecture_G=gen_net,architecture_D=dis_net,batch_size=args.batch_size,
            n_epochs=args.epochs,dataset=dataset, lr_g=args.lr_g, lr_d=args.lr_d, 
            processing_unit=processing_unit,p_subset=args.p_subset)
    else:
        raise ValueError("Invalid training function!")


