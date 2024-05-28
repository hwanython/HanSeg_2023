import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import numpy as np
import nrrd
import pandas as pd

LABEL_dict = {
    "background": 0,
    "A_Carotid_L": 1,
    "A_Carotid_R": 2,
    "Arytenoid": 3,
    "Bone_Mandible": 4,
    "Brainstem": 5,
    "BuccalMucosa": 6,
    "Cavity_Oral": 7,
    "Cochlea_L": 8,
    "Cochlea_R": 9,
    "Cricopharyngeus": 10,
    "Esophagus_S": 11,
    "Eye_AL": 12,
    "Eye_AR": 13,
    "Eye_PL": 14,
    "Eye_PR": 15,
    "Glnd_Lacrimal_L": 16,
    "Glnd_Lacrimal_R": 17,
    "Glnd_Submand_L": 18,
    "Glnd_Submand_R": 19,
    "Glnd_Thyroid": 20,
    "Glottis": 21,
    "Larynx_SG": 22,
    "Lips": 23,
    "OpticChiasm": 24,
    "OpticNrv_L": 25,
    "OpticNrv_R": 26,
    "Parotid_L": 27,
    "Parotid_R": 28,
    "Pituitary": 29,
    "SpinalCord": 30,
}

def load_nrrd(file_path):
    data, _ = nrrd.read(file_path)
    return data

def dice_score(y_true, y_pred):
    eps = 1e-6
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred)) + eps


def calculate_dice_scores(result_folder, gt_folder):
    data = []
    result_files = [f for f in os.listdir(result_folder) if f.endswith('.nrrd')]
    
    for result_file in result_files:
        case_id = result_file.split('_IMG')[0]
        gt_file = f"{case_id}_all_rois.seg.nrrd"
        
        result_path = os.path.join(result_folder, result_file)
        gt_path = os.path.join(gt_folder, gt_file)
        
        if not os.path.exists(gt_path):
            print(f"Ground truth file not found for {result_file}")
            continue
        
        result_data = load_nrrd(result_path)
        gt_data = load_nrrd(gt_path)
        
        case_dice_scores = {"file_name": result_file}
        
        for label, label_index in LABEL_dict.items():
            result_label = (result_data == label_index).astype(np.uint8)
            gt_label = (gt_data == label_index).astype(np.uint8)
            
            if np.sum(gt_label) == 0:
                case_dice_scores[label] = None
            else:
                score = dice_score(gt_label, result_label)
                case_dice_scores[label] = score
        
        # Calculate total mean DICE score for this case
        valid_scores = [score for score in case_dice_scores.values() if isinstance(score, (float, int))]
        total_dice_score = np.mean(valid_scores) if valid_scores else 0.0
        case_dice_scores["total"] = total_dice_score
        
        data.append(case_dice_scores)
        print(f"Processed {result_file}")
    
    return data


if __name__ == '__main__':
    result_folder = '/output/images/head_neck_oar'
    gt_folder = '/input/gt'

    data = calculate_dice_scores(result_folder, gt_folder)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df = df[["file_name"] + list(LABEL_dict.keys()) + ["total"]]  # Ensure columns are in the correct order
    csv_path = "/output/dice_scores.csv"
    df.to_csv(csv_path, index=False)

    print(f"CSV file saved to {csv_path}")
