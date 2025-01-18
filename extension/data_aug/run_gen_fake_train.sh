#!/bin/bash
DATASET="USC"
CLIP=0
TRAIN=1

# absolute path of the saved log path
saved_log_path="abs_path/logs/logs_USC/logdir_*"

data_path="${saved_log_path}/extracted_raw/"
python main_cls.py --train $TRAIN --dataset $DATASET --clip $CLIP --g_fake_raw_path $data_path
