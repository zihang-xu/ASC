## MICCAI2023-ASC: Appearance and Structure Consistency for Unsupervised Domain Adaptation in Fetal Brain MRI Segmentation

TODO

## Prerequisite

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh

cd **/mntnfs/med_data5/xuzihang/miccai2023**

The required packages

- ./requirements.txt
- pip install -r requirements.txt

## Data Preparation

Put the data in **./dataset**, including

- **FeTA2021 set**	
  - *./miccai2023/dataset/feta2021*
- **Atlases set**	
  - *./miccai2023/dataset/atlases*
- **Registrated set (A to F)**	
  - *./miccai2023/dataset/registrated*

## Training

python ./code/asc.py --root_path_t './dataset/feta2021' --root_path_s './dataset/atlases' --seed 1337 --consistency 200 --consistency_rampup 100

## Testing & Predictions

python test.py --root_path './dataset/feta2021' --save_mode_path './paramas/asc/iter_num_1900_dice_787.pth'

### Trained model weights

Put the trained model weights in **./params**, including

- upper
- lower
- scale
- fda
- olva
- dsa
- cutmix
- asenet
- asc (ours)

