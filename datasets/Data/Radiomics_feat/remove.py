import os

file_path  ='/inspire/hdd/project/continuinglearning/suhaoyang-240107100018/suhaoyang-240107100018/storage/RU-net/datasets/Data/Radiomics_feat/BCLM_PLC_HHM/5-fold_0915'

for item in os.listdir(file_path):
    if 'scaled' not in str(item) and 'txt' not in str(item):
        os.remove(f"{file_path}/{item}")