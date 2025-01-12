DATASET="wifi"
CLIP=0
TRAIN=1
split_num=4

if [[ "$DATASET" == "USC" || "$DATASET" == "pamap" ]]; then
    split_num=4
elif [[ "$DATASET" == "mmwave" || "$DATASET" == "wifi" ]]; then
    split_num=5
else
    echo "Unknown DATASET value"
    exit 1
fi

get_absolute_path() {
    local relative_path="$1"
    local absolute_path

    if [[ -z "$relative_path" ]]; then
        echo "Error: Relative path is not provided."
        return 1
    fi

    absolute_path=$(realpath "$relative_path")
    echo "$absolute_path"
}

data_path="../../src/data_utils/saved/${DATASET}/split"
abs_data_path=$(get_absolute_path "$data_path")


cd ../../src/baseline_dir/clswagan

for split in $(seq 1 ${split_num}); do
    echo "Run Data Augmentation Split ${split}"
    split_data_path="${abs_data_path}${split}/Type_BASE_${DATASET}/extracted_raw/"
    python main_cls.py --train $TRAIN --dataset $DATASET --clip $CLIP --g_fake_raw_path $split_data_path --split $split
done