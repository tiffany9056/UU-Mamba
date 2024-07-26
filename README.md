# UU-Mamba
Ting Yu Tsai, Li Lin, Shu Hu, Ming-Ching Chang, Hongtu Zhu, Xin Wang

This repository is the official implementation of our paper [UU-Mamba: Uncertainty-aware U-Mamba for Cardiac Image Segmentation](https://arxiv.org/pdf/2405.17496).

We introduce a new model UU-Mamba for segmenting cardiac MRI images. The model combines the U-Mamba model, an uncertainty-aware loss function, and the SAM optimizer.
_________________

## Installation
Requirements: `Ubuntu: 20.04`, `CUDA 11.8`
1. Install Pytorch 2.0.1:
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```
2. Install [Mamba](https://github.com/state-spaces/mamba):
```bash
pip install causal-conv1d>=1.2.0 and pip install mamba-ssm --no-cache-dir
```
3. Download code:
```bash
git clone https://github.com/tiffany9056/UU-Mamba.git
```
4. cd U-Mamba/umamba and run pip install -e .
```bash
cd UU-Mamba/uumamba
pip install -e
```
https://github.com/tiffany9056/UU-Mamba/assets/39300518/5fcceb3a-0820-47a4-aeac-a83368102fd1

## Model Training
Download dataset [here](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) and put them into the `data` folder.
Since U-Mamba is build on the `nnU-Net` framework, they have updated some code in the new version, you need to update the new version using this [website](https://github.com/MIC-DKFZ/nnUNet/commit/f569e34d0265723288a64eca579609e0274b1a0b).

### Set environment variables
The default data directory for UU-Mamba is preset to UU-Mamba/data. Users with existing nnU-Net setups who wish to use alternative directories for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` can easily adjust these paths in UU-Mamba/umamba/nnunetv2/path.py to update your specific nnUNet data directory locations, as demonstrated below:
```bash
export nnUNet_raw="/UU-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/UU-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/UU-Mamba/data/nnUNet_results"
```
or
```python
# An example to set other data path,
base = '/UU-Mamba/data' # or you can set your own path, e.g., base = '/home/user_name/Documents/UU-Mamba/data'
nnUNet_raw = join(base, 'nnUNet_raw') # or change to os.environ.get('nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # or change to os.environ.get('nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results') # or change to os.environ.get('nnUNet_results')
```
Verify that environment parameters are set: execute  `echo ${nnUNet_raw}` etc to print the environment variables. This will return an empty string if they were not set.

### Preprocessing
1. Replace the new file of ACDC dataset conversion. The new file is located [here](https://drive.google.com/drive/folders/1AQTtWgYsxX9KC_Xn8PApRKOfJnnowtBa?usp=sharing). You need to execute the following lines (adapt the paths to the actual folders you intend to use).
```bash
cp /UU-Mamba/ACDC_code/Dataset027_ACDC.py /UU-Mamba/uumamba/nnunetv2/dataset_conversion/Dataset027_ACDC.py
```
2. Run the dataset conversion file `Dataset027_ACDC.py`.
```bash
python /UU-Mamba/uumamba/nnunetv2/dataset_conversion/Dataset027_ACDC.py -i /data/ACDC/database
```
3. Preprocess the data runing `nnUNetv2_plan_and_preprocess`, ACDC dataset ID `DATASET_ID` is 027.
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Train model
ACDC dataset ID `DATASET_ID` is 027.
```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaEnc
```
If you want to use pretrained model, download pretrain weight [here](https://drive.google.com/drive/folders/1AQTtWgYsxX9KC_Xn8PApRKOfJnnowtBa?usp=sharing) and put it into `pretrain_weight` folder, then run
```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaEnc -pretrained_weights /UU-Mamba/pretrain_weight/checkpoint_UU-Mamba.pth
```

## Inference
Predict testing cases with model
```text
Example to parameter setting:
INPUT_FOLDER  -> /UU-Mamba/data/nnUNet_raw/Dataset027_ACDC/imagesTs/
OUTPUT_FOLDER -> /results/
DATASET_ID    -> 027
CHK_NAME      -> checkpoint_best.pth or checkpoint_latest.pth or checkpoint_final.pth ...
GT_LABEL      -> /UU-Mamba/data/nnUNet_raw/Dataset111_IRIS/labelsTs
EVL_FOLDER    -> /results/
```
```bash
nnUNetv2_predict -i INPUT_FOLDER -o /results/seg/niigz -d DATASET_ID -c 3d_fullres -f all -tr nnUNetTrainerUMambaEnc --disable_tta -chk CHK_NAME
```
After predict testing cases with model, run IRIS_DSC_Eval.py to have the result DSC of each testing cases in csv file.
```text
Example to parameter setting:
GT_LABEL   -> /UU-Mamba/data/nnUNet_raw/Dataset111_IRIS/labelsTs # Testing ground truth data path.
SEG_FOLDER -> /results/ # The output segmentation testing data path. (usually is the same as OUTPUT_FOLDER)
EVL_FOLDER -> /results/ # Where result csv file save.
```
```bash
python /UU-Mamba/evaluation/ACDC_DSC_Eval.py --gt_path GT_LABEL --seg_path SEG_FOLDER --save_path EVL_FOLDER
```

## Citation
Please kindly consider citing our papers in your publications. 
```bash
@misc{tsai2024uumamba,
      title={UU-Mamba: Uncertainty-aware U-Mamba for Cardiac Image Segmentation}, 
      author={Ting Yu Tsai and Li Lin and Shu Hu and Ming-Ching Chang and Hongtu Zhu and Xin Wang},
      year={2024},
      eprint={2405.17496},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
@article{wang2024artificial,
  title={Artificial Intelligence in Image-based Cardiovascular Disease Analysis: A Comprehensive Survey and Future Outlook},
  author={Wang, Xin and Zhu, Hongtu},
  journal={arXiv:2402.03394},
  year={2024}
}
```
