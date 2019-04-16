import torch

from imageio import imread, imsave
from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import DispNetS
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")     #choose to resize image or not

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")     #use CUDA if possible otherwise use CPU
print(device)

@torch.no_grad()        #temporarily set all the requires_grad flag to false
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):          #cmd must choose disp or depth
        print('You must at least output one value !')
        return

    disp_net = DispNetS().to(device)                        #load network to cuda or cpu device
    weights = torch.load(args.pretrained)                   #weights = pretrained model
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:                                                           #if there's something in dataset_list file
        with open(args.dataset_list, 'r') as f:                                                 #read dataset_list file
            test_files = [dataset_dir/file for file in f.read().splitlines()]                   #test_files = dataset_list
    else:
        test_files = sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])  #otherwise test_files = files in dataset_dir

    print('{} files to test'.format(len(test_files)))                                       #print test_files (images to execute on)

    for file in tqdm(test_files):                                                           #for every test_file
        print(file) ###
        img = imread(file).astype(np.float32)                                               #img = test_file

        h,w,_ = img.shape                                                                   #calculate height and width
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):          #img = resized test_file if needed
            img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))                                                  #swaps axis 0,1,2 positions

        tensor_img = torch.from_numpy(img).unsqueeze(0)                                     #creates tensor from numpy array, add singleton dimension
        tensor_img = ((tensor_img/255 - 0.5)/0.2).to(device)                                #

        output = disp_net(tensor_img)[0]                                                    #run images through model: OUTPUT

        ### OUTPUT output VALUES IN TEXT FILE
        depth = 1/output
        output_np = output.detach().cpu().numpy()

        with open('depth_numpy_pt.txt','w') as f:
            for r in range(args.img_height):
                print(output_np.shape)
                print(str(r) + ', 0:' + str(args.img_width ))
                f.write('\nROW ' + str(r) + ', COL 0:' + str(args.img_width) +  '\n')
                f.write(str(output_np[0,r,:]))
        ###

        file_path, file_ext = file.relpath(args.dataset_dir).splitext()                     #name output files
        file_name = '-'.join(file_path.splitall())

        if args.output_disp:                                                                            #save disparity
            disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            disp = disp[:3,:,:]
            imsave(output_dir/'{}_disp{}'.format(file_name, file_ext), np.transpose(disp, (1,2,0)))
        if args.output_depth:                                                                           #save depth
            depth = 1/output
            depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
            #a= depth.shape  #(4,128,416)
            #print(a)
            depth = depth[:3,:,:]
            imsave(output_dir/'{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1,2,0)))


if __name__ == '__main__':
    main()

