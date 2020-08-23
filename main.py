import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Silences unnecessary spam from TensorFlow libraries. Set to 0 for full output

from helper_functions import parse_arguments
from helper_functions import set_gpus

def main(args):
    set_gpus(n_gpus=args.n_gpus, min_vram=args.min_vram, split_gpu_into=args.split_gpu_into)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
