import os

# Best: 0.352847158908844
sh = "python advent.py --dataset_root /home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np" \
     " -s MR -t CT --batch-size 8 --seg_loss Focal --gpu_ids 0" \
     " --train-size 512 512 --test-input-size 512 512 --test-output-size 512 512" \
     " --log logs/advent/synthia2cityscapes -i 1000"

os.system(sh)
