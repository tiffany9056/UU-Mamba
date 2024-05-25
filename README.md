# UU-Mamba
We introduce a new model UU-Mamba for segmenting cardiac MRI images. The model combines the U-Mamba model, an uncertainty-aware loss function, and the SAM optimizer.
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
git clone https://github.com/bowang-lab/U-Mamba
```
4. cd U-Mamba/umamba and run pip install -e .
```bash
cd U-Mamba/umamba
pip install -e
```
## Model Training
Download dataset [here](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) and put them into the `data` folder.
Since U-Mamba is build on the `nnU-Net` framework, they have updated some code in the new version, you need to update the new version using this [website](https://github.com/MIC-DKFZ/nnUNet/commit/f569e34d0265723288a64eca579609e0274b1a0b).
## Set environment variables
The default data directory for UU-Mamba is preset to U-Mamba/data. Users with existing nnU-Net setups who wish to use alternative directories for `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` can easily adjust these paths in U-Mamba/umamba/nnunetv2/path.py to update your specific nnUNet data directory locations, as demonstrated below:
```bash
export nnUNet_raw="/workspace/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/workspace/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/workspace/U-Mamba/data/nnUNet_results"
```
or
```python
# An example to set other data path,
base = '/workspace/U-Mamba/data'# or you can set your own path, e.g., base = '/home/user_name/Documents/U-Mamba/data'
nnUNet_raw = join(base, 'nnUNet_raw') # or change to os.environ.get('nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') # or change to os.environ.get('nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results') # or change to os.environ.get('nnUNet_results')
```
Verify that environment parameters are set: execute  `echo ${nnUNet_raw}` etc to print the environment variables. This will return an empty string if they were not set.

## Preprocessing
1. replace the new file of ACDC dataset conversion. The new file is located in folder [ACDC_code](). You need to execute the following lines (adapt the paths to the actual folders you intend to use). Then, run the dataset conversion file `Dataset027_ACDC.py`. Also preprocess the data runing `nnUNetv2_plan_and_preprocess`, ACDC dataset ID `DATASET_ID` is 027:
```bash
cp /ACDC_code/Dataset027_ACDC.py /workspace/U-Mamba/umamba/nnunetv2/dataset_conversion/Dataset027_ACDC.py
python /workspace/U-Mamba/umamba/nnunetv2/dataset_conversion/Dataset027_ACDC.py -i /data/ACDC/database
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

```
2. Add the evaluation file into `U-Mamba/evaluation/`. You need to execute the following lines (adapt the paths to the actual folders you intend to use):
```bash
cp /ACDC_code/ACDC_DSC_Eval.py /workspace/U-Mamba/evaluation/ACDC_DSC_Eval.py
```


## Data Preparation
- **ACDC Original Dataset**:
   - Download the ACDC original dataset from [ACDC_original_dataset](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb).
   - Place the downloaded files into the `ACDC_original_dataset` folder.
