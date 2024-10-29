# reschedule the data scheme of validation seen and unseen data
import random

def mix_val_data(val_seen_meta, val_unseen_meta, method=0, equal_num=None):
    """
    method: 0 -> equal seen and unseen
    method: 1 -> the number of seen and unseen follows the ratio of the number of classes
    method: 2 -> reduce both seen and unseen to the same number of samples
    """
    val_seen_samples_num = val_seen_meta["samples"]
    val_unseen_samples_num = val_unseen_meta["samples"]

    mix_val_meta = {"type": "mix_val", "samples": 0, "seen_label_text_dict": val_seen_meta["label_text_dict"],
                    "unseen_label_text_dict":  val_unseen_meta["label_text_dict"], "data_list": []}
    random.shuffle(val_unseen_meta["data_list"])
    random.shuffle(val_seen_meta["data_list"])
    if method == 0:
        if val_seen_samples_num == val_unseen_samples_num:
            pass
        elif val_unseen_samples_num > val_seen_samples_num:
            # TODO: randomly build dataset (X)
            # TODO: Recommend to use the same number of samples in seen and unseen classes (V)
            mix_val_meta["data_list"] = val_seen_meta["data_list"] + val_unseen_meta["data_list"][:val_seen_samples_num]
            mix_val_meta["samples"] = len(mix_val_meta["data_list"])
        else:
            mix_val_meta["data_list"] = val_unseen_meta["data_list"] + val_seen_meta["data_list"][:val_unseen_samples_num]
            mix_val_meta["samples"] = len(mix_val_meta["data_list"])
    elif method == 1:
        seen_classes_num = len(val_seen_meta["label_text_dict"])
        unseen_classes_num = len(val_unseen_meta["label_text_dict"])

        assert seen_classes_num > unseen_classes_num
        assert val_seen_samples_num > val_unseen_samples_num

        class_ratio_num = int(val_seen_samples_num * (unseen_classes_num / seen_classes_num))
        mix_val_meta["data_list"] = val_seen_meta["data_list"] + val_unseen_meta["data_list"][:class_ratio_num]
    elif method == 2:
        if not equal_num:
            equal_num = 5000
        mix_val_meta["data_list"] = val_seen_meta["data_list"][:equal_num] + val_unseen_meta["data_list"][:equal_num]
        mix_val_meta["samples"] = len(mix_val_meta["data_list"])
    elif method == 3:
        if val_seen_samples_num == val_unseen_samples_num:
            pass
        elif val_unseen_samples_num > val_seen_samples_num:
            val_unseen_meta["data_list"] = val_unseen_meta["data_list"][:val_seen_samples_num]
        elif val_seen_samples_num > val_unseen_samples_num:
            val_seen_meta["data_list"] = val_seen_meta["data_list"][:val_unseen_samples_num]
        mix_val_meta["data_list"] = val_seen_meta["data_list"] + val_unseen_meta["data_list"]
        return val_seen_meta, val_unseen_meta, mix_val_meta
    elif method == 4:
        # balance each classes num
        seen_class_dict = {}
        unseen_class_dict = {}

        seen_data_num = len(val_seen_meta["data_list"])
        unseen_data_num = len(val_unseen_meta["data_list"])

        for data in val_seen_meta["data_list"]:
            if data["label"] not in seen_class_dict:
                seen_class_dict[data["label"]] = []
            seen_class_dict[data["label"]].append(data)

        for data in val_unseen_meta["data_list"]:
            if data["label"] not in unseen_class_dict:
                unseen_class_dict[data["label"]] = []
            unseen_class_dict[data["label"]].append(data)

        seen_class_num = len(seen_class_dict)
        unseen_class_num = len(unseen_class_dict)

        if seen_data_num >= unseen_data_num:
            mix_val_meta["data_list"].extend(val_unseen_meta["data_list"])
            each_label_num = unseen_data_num // seen_class_num
            res_num = unseen_data_num
            for label in seen_class_dict:
                if len(seen_class_dict[label]) >= each_label_num:
                    mix_val_meta["data_list"].extend(seen_class_dict[label][:each_label_num])
                    res_num -= each_label_num
                else:
                    mix_val_meta["data_list"].extend(seen_class_dict[label])
                    res_num -= len(seen_class_dict[label])
            for label in seen_class_dict:
                if res_num == 0:
                    break
                if len(seen_class_dict[label][each_label_num:]) >= res_num:
                    mix_val_meta["data_list"].extend(seen_class_dict[label][each_label_num: each_label_num + res_num])
                    break
                else:
                    mix_val_meta["data_list"].extend(seen_class_dict[label][each_label_num:])
                    res_num -= len(seen_class_dict[label][each_label_num:])
        else:
            mix_val_meta["data_list"].extend(val_seen_meta["data_list"])
            each_label_num = seen_data_num // unseen_class_num
            res_num = seen_data_num
            for label in unseen_class_dict:
                if len(unseen_class_dict[label]) >= each_label_num:
                    mix_val_meta["data_list"].extend(unseen_class_dict[label][:each_label_num])
                    res_num -= each_label_num
                else:
                    mix_val_meta["data_list"].extend(unseen_class_dict[label])
                    res_num -= len(unseen_class_dict[label])
            for label in unseen_class_dict:
                if res_num == 0:
                    break
                if len(unseen_class_dict[label][each_label_num:]) >= res_num:
                    mix_val_meta["data_list"].extend(unseen_class_dict[label][each_label_num: each_label_num + res_num])
                    break
                else:
                    mix_val_meta["data_list"].extend(unseen_class_dict[label][each_label_num:])
                    res_num -= len(unseen_class_dict[label][each_label_num:])
    else:
        raise ValueError("Invalid method")

    random.shuffle(mix_val_meta["data_list"])
    mix_val_meta["samples"] = len(mix_val_meta["data_list"])
    return mix_val_meta

def validation_seen_dataset():
    pass


if __name__ == "__main__":
    mix_val_data()