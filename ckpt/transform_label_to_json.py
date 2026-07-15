import os
import pandas as pd
import json

csv_info = pd.read_csv('./download/merlinabdominalctdataset/zero_shot_findings_disease_cls.csv')
all_column = list(csv_info.columns)

all_labels = {}
pids = csv_info[all_column[0]]
for disease in all_column[1:]:
    print(disease)
    dlabel = csv_info[disease]
    pid_dlabel = {pids[i]:int(dlabel[i]) for i in range(len(pids))}
    all_labels[disease] = pid_dlabel

save_path = './ckpt/merlin_labels.json'
json.dump(all_labels, open(save_path, 'w'), ensure_ascii=False, indent=4)

print('done')



