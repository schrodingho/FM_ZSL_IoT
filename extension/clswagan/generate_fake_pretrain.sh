DATASET="USC"
read_path='/home/dingding/PycharmProjects/Efficient-Prompt/logs_USC/logdir_20240410-231245_another_best/extracted_raw/'
CLIP=0
TRAIN=1

python main_cls.py --train $TRAIN --dataset $DATASET --clip $CLIP --g_fake_raw_path $read_path

