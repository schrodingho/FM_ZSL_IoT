# all splits
DATASET="wifi"
baseline_arg=0
test_on=True
split_num=5
ablation="_has_prompt_has_data_aug_1"
# has_data_aug

if [[ "$DATASET" == "USC" || "$DATASET" == "pamap" ]]; then
    split_num=4
elif [[ "$DATASET" == "mmwave" || "$DATASET" == "wifi" ]]; then
    split_num=5
else
    echo "Unknown DATASET value"
    exit 1
fi

cd ../src


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

for split in $(seq 1 ${split_num});
do
    echo "Run Split $split"
    back_up_path="./data_utils/saved/${DATASET}/split$split/Type_BASE_${DATASET}"
    absolute_path=$(get_absolute_path "$back_up_path")

    python main_entry.py --config_choose $DATASET --back_up_path $absolute_path --baseline_arg $baseline_arg --test_on $test_on --ablation $ablation
done