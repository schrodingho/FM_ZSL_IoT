#!/bin/bash
DATASET="USC"
CLIP=0
TRAIN=1

saved_log_path="/home/dingding/PycharmProjects/FM_ZSL_IoT/logs/logs_USC/logdir_20250113-224430"

split_data_path="${abs_data_path}${split}/Type_BASE_${DATASET}/extracted_raw/"
python main_cls.py --train $TRAIN --dataset $DATASET --clip $CLIP --g_fake_raw_path $saved_log_path
