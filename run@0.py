import os

sh = "python cycle_gan.py --dataset_root /home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np" \
     " -s Synthia -t Cityscapes --batch-size 8 --gpu_ids 0" \
     " --log logs/advent/synthia2cityscapes"

os.system(sh)
