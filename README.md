## MICCAI2023-ASC: Appearance and Structure Consistency for Unsupervised Domain Adaptation in Fetal Brain MRI Segmentation

TODO

## Prerequisite

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh

cd **/mntnfs/med_data5/xuzihang/miccai2023**

please change to your own path.

The required packages

- ./requirements.txt
- pip install -r requirements.txt

## Data Preparation

Put the data in **./dataset**, including

- **FeTA2021 set**	
  - *./miccai2023/dataset/feta2021*
  - https://feta.grand-challenge.org/feta-2021/
  - or you can find dataset on this web: https://zenodo.org/records/4541606
- **Atlases set**	
  - *./miccai2023/dataset/atlases*
  - [1] Gholipour, Ali, et al. "A normative spatiotemporal MRI atlas of the fetal brain for automatic segmentation and analysis of early brain growth." Scientific reports 7.1 (2017): 1-13.
  - [2] Wu, Jiangjie, et al. "Age-specific structural fetal brain atlases construction and cortical development quantification for chinese population." Neuroimage 241 (2021): 118412.
  - [3] Fidon, Lucas, et al. "A spatio-temporal atlas of the developing fetal brain with spina bifida aperta." Open Research Europe (2021).
  - or you can cite this repository: https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation/tree/master/data
- **Registrated set (A to F)**	
  - *./miccai2023/dataset/registrated*
  
The first two data sets are publicly available

## Training

python ./code/asc.py --root_path_t './dataset/feta2021' --root_path_s './dataset/atlases' --seed 1337 --consistency 200 --consistency_rampup 100

## Testing & Predictions

python test.py --root_path './dataset/feta2021' --save_mode_path './paramas/asc/iter_num_1900_dice_787.pth'

The model parameter can be obtained in "parameter"

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

