DATASET="USC"
CLIP=1
TRAIN=1
read_path='/home/dingding/PycharmProjects/Efficient-Prompt/logs_USC/logdir_20240407-211928/extracted_raw/'
python main_cls.py --train $TRAIN --dataset $DATASET --clip $CLIP --g_fake_attr_path $read_path

