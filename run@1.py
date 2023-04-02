import os

# DiceFocal x
sh = "python advent.py --dataset_root /home/yht/Casit/Datasets/ez/datasets/MMWHS/data_np" \
     " -s MR -t CT --seg_loss GeneralizedDice --batch-size 16 --gpu_ids 1" \
     " --log logs/advent/MMWHS_GDiceFocal"

os.system(sh)
