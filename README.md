# Leveraging Foundation Models for Zero-Shot IoT Sensing
[![](https://img.shields.io/badge/LICENSE-MIT-blue?style=flat)](https://github.com/schrodingho/FM_ZSL_IoT/blob/main/LICENSE) 
[![](https://img.shields.io/badge/ECAI-2024-purple?style=flat)](https://www.ecai2024.eu/calls/main-track) 
[![](https://img.shields.io/badge/arXiv:2407.19893-red?style=flat)](https://arxiv.org/pdf/2407.19893)

## Abstract
Deep learning models are increasingly deployed on edge Internet of Things (IoT) devices. However, these models typically operate under supervised conditions and fail to recognize unseen classes different from training. To address this, zero-shot learning (ZSL) aims to classify data of unseen classes with the help of semantic information. Foundation models (FMs) trained on web-scale data have shown impressive ZSL capability in natural language processing and visual understanding. However, leveraging FMs' generalized knowledge for zero-shot IoT sensing using signals such as mmWave, IMU, and Wi-Fi has not been fully investigated. In this work, we align the IoT data embeddings with the semantic embeddings generated by an FM's text encoder for zero-shot IoT sensing. To utilize the physics principles governing the generation of IoT sensor signals to derive more effective prompts for semantic embedding extraction, we propose to use cross-attention to combine a learnable soft prompt that is optimized automatically on training data and an auxiliary hard prompt that encodes domain knowledge of the IoT sensing task. To address the problem of IoT embeddings biasing to seen classes due to the lack of unseen class data during training, we propose using data augmentation to synthesize unseen class IoT data for fine-tuning the IoT feature extractor and embedding projector. We evaluate our approach on multiple IoT sensing tasks. Results show that our approach achieves superior open-set detection and generalized zero-shot learning performance compared with various baselines.

## Setup & Usage
### Environment setup (Linux)
- Create and activate the python virtual environment. [(Tips)](https://stackoverflow.com/questions/43069780/how-to-create-virtual-env-with-python3)
- Install all requirements: `pip install -r requirements.txt`
### Data preparation
- [USC-HAD](https://sipi.usc.edu/had/): Change the `dataset_path` in `./settings/USC.yaml` as your USC-HAD directory path (delete the `*.m` and `*.txt` in the directory's first level menu in advance).
- [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring): Get the path of the subdirectory `Protocol` of PAMAP2 dataset, change the `dataset_path` in `./settings/pamap.yaml` as your own `Protocol` path.
- [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi)
  - [mmWave](https://drive.google.com/file/d/1KxPaB2amj0mQkjhrx_1yfPQ0_s2H58tx/view?usp=drive_link): We use the official `filtered_mmwave` for training and testing. Change the `dataset_path` in `./settings/mmwave.yaml` as your `filtered_mmwave` directory path.
  - [Wi-Fi](https://github.com/ybhbingo/MMFi_dataset): We use the official Wi-Fi dataset from MM-Fi. The [E04](https://drive.google.com/file/d/1-XTwxO0ymJ1AtI5HsOOjD-XTrIHKPaA1/view?usp=drive_link) is selected for training and testing. Change the `dataset_path` in `./settings/wifi.yaml` as your Wi-Fi dataset directory path.
### Training
- quick train
```python
python main.py --config_choose <dataset_config>
```
- train on previous saved log/data
```python
python main.py --config_choose <dataset_config> --back_up_path <path_to_saved_log_or_data>
```
- view logs: check the logs in the `./logs`
### Inference
- only foundation model
```python
python main.py --config_choose <dataset_config> --back_up_path <path_to_saved_log_or_data> --test_model_path <path_to_saved_model>
```

- local supervised model + foundation model
```python
# Train local model first (back_up_path use the same data as the previous trained foundation model)
python main_sup.py --config_choose <dataset_config> --back_up_path <path_to_saved_log_or_data>

# Inference local + foundation model
python main.py --config_choose <dataset_config> --back_up_path <path_to_saved_log_or_data> --test_model_path <path_to_saved_fm_model> --local_model_path <path_to_saved_local_model>
```

- data augmentation
```python
# Generate augmented data (a trained model with its log is requried)
cd ./extension/data_aug
# change the `saved_log_path` in `run_gen_fake_train.sh` as the path to the saved log
# run the script
bash run_gen_fake_train.sh
```

```python
# train with augmented data
python main.py --config_choose <dataset_config> --back_up_path <path_to_saved_log_or_data> --fake True
```

## Thanks
- [CLIP](https://github.com/openai/CLIP)
- [Efficient-Prompt](https://github.com/ju-chen/Efficient-Prompt)
- [SupContrast](https://github.com/HobbitLong/SupContrast)
- [HAR-Dataset-Preprocess](https://github.com/xushige/HAR-Dataset-Preprocess)
- [f-clswgan-pytorch](https://github.com/mkara44/f-clswgan_pytorch)