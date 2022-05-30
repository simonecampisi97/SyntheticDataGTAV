import os

import pandas as pd
import tqdm, json

dict_ann = {

    "frame": 0,
    "pedestrian_id": 1,
    "joint_type": 2,
    "2D_y": 3,
    "3D_x": 4,
    "3D_y": 5,
    "3D_z": 6,
    "occluded": 7,
    "cam_3D_z": 8,
    "cam_rot_x": 9,
    "cam_rot_y": 10,
    "cam_rot_z": 11,
    "fov": 12,
}

JTA_dataset_cols = ['frame', 'pedestrian_id', 'joint_type', '2D_x', '2D_y', '3D_x', '3D_y',
                    '3D_z', 'occluded', 'self_occluded']

if __name__ == "__main__":

    # Conversation to JTA DATASET JASON

    for j in range(2, 14):
        seq_path = f"C:\\Users\\simoc\\Desktop\\Synthetic Data IMAVIS\\seq_{j}"

        print(seq_path)
        df = pd.read_csv(os.path.join(seq_path, "coords.csv"))

        ann = []

        for i, row in tqdm.tqdm(enumerate(df.iterrows())):
            ann.append(row[1][JTA_dataset_cols].tolist())

        with open(os.path.join(seq_path, f"seq_{j}.json"), "w") as f:
            json.dump(ann, f)
