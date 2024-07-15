# Overcoming the Pitfalls of Vision-Language Model for Image-Text Retrieval


## Setup
```bash
# Create python environment (optional)
conda create -n LG-MGC python=3.8
source activate LG-MGC

# Install python dependencies
pip install -r requirements.txt


```

## Code structure
```bash
./data
./datasets
  Coco.py
  Flickr30k.py
  ...
# Training and testing in the LG-MGC setting
./model
    build_model.py                         <= Our LG-MGC model classes
    ...
./processor
    processor.py                           <= Training  in the LG-MGC setting
./solver
./utils
build_dalle.py                             <= Load diffusion model
train.py                                   <= Training entrance
test.py                                    <= Testing entrance
    

```

## Dataset Preparation / Model checkpoint
- Download COCO and Flickr30k datasets from the original websites
- Download `annotation`,from 
- https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json
- https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json
- Download other JSON files and replace train with test and val.
- Download diffusion model checkpoints from https://huggingface.co/laion/DALLE2-PyTorch/tree/main/prior

## Tasks

```bash
# Training 
python train.py --name LOG_NAME --img_aug --num_epoch 6 --batch_size 32 --loss_names 'itc+pred+top+topsm' --dataset_name 'coco'
python train.py --name LOG_NAME --img_aug --num_epoch 6 --batch_size 32 --loss_names 'itc+pred+top+topsm' --dataset_name 'Flickr30k'
# Testing
python test.py
```
