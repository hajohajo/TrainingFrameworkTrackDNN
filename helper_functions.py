from argparse import ArgumentParser
from os import environ
from gpustat import GPUStatCollection
from numpy import argsort
from tensorflow import config

'''
Parser for command line arguments
--n_gpus: number of physical GPUs to assign for the training (defaults to 1). Set to 0 for CPU training.
--min_vram: minimum amount of free vram required for running the model training on a GPU/GPU partition (defaults to 2000MB)
--split_gpu_into: into how many logical GPUs each physical GPU should be partitioned. Note that each physical GPU then needs to have min_vram * split_gpu_into MB free VRAM. (defaults to 1).
                  This speeds up training as it allows distributed training to be used on the GPU(s).
'''
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=1,
                        help="How many GPUs to utilize (default 1)")
    parser.add_argument('--min_vram', type=float, default=2000.0,
                        help="How much VRAM needed to load and train the model on GPU")
    parser.add_argument('--split_gpu_into', type=int, default=1,
                        help="Split each physical GPU to this many logical GPUs to speedup training.")
    args = parser.parse_args()


    if args.n_gpus < 0:
        parser.error(f"Invalid argument --n_gpus={args.n_gpus}. Use non-negative integers.")
    if args.min_vram < 0.0:
        parser.error(f"Invalid argument --min_vram={args.min_vram}. Use positive amount of memory.")
    if args.split_gpu_into < 1:
        parser.error(f"Invarlid argument --split_gpu_into={args.split_gpu_into}. Use a positive integer.")

    return args

'''
Configures the GPUs to be allocated for training, preferring the GPUs with most free VRAM.

n_gpus: How many physical GPUs to allocate for this training process. Set to 0 to run on CPU.
min_memory: How much free VRAM each physical GPU has to have. Too low value causes an error if the GPU runs out of memory when training.
split_into: How many logical GPUs to split each physical GPU into. This can speed up the training due to distributed training.
            Each physical GPU has to have min_memory * split_into VRAM available or an error is raised.

'''
def set_gpus(n_gpus=1, min_vram=2000, split_gpu_into=1):
    gpu_stats = GPUStatCollection.new_query()
    gpu_ids = map(lambda gpu: int(gpu.entry['index']), gpu_stats)
    gpu_freemem = map(lambda gpu: float(gpu.entry['memory.total']-gpu.entry['memory.used']), gpu_stats)
    pairs = list(zip(gpu_ids, gpu_freemem))
    valid_pairs = [pair for pair in pairs if pair[1] >= min_vram * split_gpu_into]

    if len(valid_pairs) < n_gpus:
        raise ValueError(f"Not enough valid GPUs detected. Check if the machine has at least {n_gpus} GPUs with at least {min_vram * split_gpu_into}MB free VRAM or set a lower --n_gpus value")

    sorted_indices = list(argsort([mem[1] for mem in valid_pairs]))[::-1]
    sorted_pairs = [valid_pairs[i] for i in sorted_indices]
    print(f"Setting {n_gpus} physical GPUs split into {n_gpus * split_gpu_into} logical GPUs with {min_vram}MB VRAM each for this training")
    environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    devices = ",".join([str(pair[0]) for pair in sorted_pairs[:n_gpus]])
    environ['CUDA_VISIBLE_DEVICES'] = devices

    if split_gpu_into > 1:
        physical_devices = config.list_physical_devices('GPU')
        for device in physical_devices:
            config.set_logical_device_configuration(
                device,
                [config.LogicalDeviceConfiguration(memory_limit=min_vram) for _ in range(split_gpu_into)]
            )
