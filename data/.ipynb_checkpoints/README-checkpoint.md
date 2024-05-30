Download ACDC dataset [here](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb)

Please organize the dataset as follows:

```
data/  
├── nnUNet_raw/
│   ├── Dataset027_ACDC/
│   │   ├── imagesTr
│   │   │   ├── patient001_frame01_0000.nii.gz
│   │   │   ├── patient001_frame12_0000.nii.gz
│   │   │   ├── ...
│   │   ├── imagesTs
│   │   │   ├── patient101_frame01_0000.nii.gz
│   │   │   ├── patient101_frame14_0000.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTr
│   │   │   ├── patient001_frame01.nii.gz
│   │   │   ├── patient001_frame12.nii.gz
│   │   │   ├── ...
│   │   ├── labelsTs
│   │   │   ├── patient101_frame01.nii.gz
│   │   │   ├── patient101_frame14.nii.gz
│   │   │   ├── ...
│   │   ├── dataset.json

```