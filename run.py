import os

sh = "python advent.py --dataset_root /home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np" \
     " -s Synthia -t Cityscapes --batch-size 8" \
     " --log logs/advent/synthia2cityscapes"

os.system(sh)