import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import argparse

join = os.path.join
basename = os.path.basename

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type = str, required = True,
                    help = 'testing ground truth data path')
parser.add_argument('--seg_path', type = str, required = True,
                    help = 'The output segmentation testing data path')
parser.add_argument('--save_path', type = str, required = True,
                    help = 'Result csv file output folder.')
args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
label_tolerance = OrderedDict({'RV': 1, 'MLV':2, 'LVC':3})
for organ in label_tolerance.keys():
    seg_metrics['{}_DSC'.format(organ)] = list()

def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.

    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int

    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) ==1, print('mask label error!')
    z_index = np.where(organ_mask>0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)
    
    return z_lower, z_upper

for name in tqdm(filenames):
    seg_metrics['Name'].append(name)
    # load grond truth and segmentation
    gt_nii = nb.load(join(gt_path, name))
    case_spacing = gt_nii.header.get_zooms()
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())
    
    for i, organ in enumerate(label_tolerance.keys(),1):
        if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
            DSC_i = 0
            NSD_i = 0
        else:
            organ_i_gt, organ_i_seg = gt_data==i, seg_data==i          
            DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
            
        seg_metrics['{}_DSC'.format(organ)].append(round(DSC_i, 4))

    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(save_path + 'result_DSC.csv', index=False)

case_avg_DSC = dataframe.mean(axis=0, numeric_only=True)
print(20 * '>')
print(f'Average DSC for {basename(seg_path)}: {case_avg_DSC.mean()}')
print(20 * '<')
