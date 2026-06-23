import numpy as np 
import pandas as pd 
import os 

def generate_stage3_numerics():

    input_path = os.path.join(".", "stage2_synthetic_data.csv")
    output_path = os.path.join(".", "stage3_synthetic_data.csv")

    if not os.path.exists(input_path):
        print(f"[-] Error: Missing baseline data at {input_path}")
        print("Please ensure your stage2 file is inside data/datasets/")
        return
    
    df = pd.read_csv(input_path)
    num_rows = len(df)
    np.random.seed(42)

    hippo_vol = np.zeros(num_rows, dtype=np.float32)
    ventricle_vol = np.zeros(num_rows, dtype=np.float32)
    gray_matter_vol = np.zeros(num_rows, dtype=np.float32)

    print(f"[+] Processing {num_rows} patient profiles...")

    for idx, row in df.iterrows():
        dx = int(row["diagnostic_class"])
        age = row["age"]

        if dx == 0:    # Cognitively Normal
            hippo = np.random.normal(3685, 400) - (age - 65) * 10
            ventricle = np.random.normal(38300, 12000) + (age - 65) * 200
            gm = np.random.normal(600, 40) - (age - 65) * 1.5
        elif dx == 1:  # Mild Cognitive Impairment
            hippo = np.random.normal(3166, 450) - (age - 65) * 15
            ventricle = np.random.normal(45800, 15000) + (age - 65) * 350
            gm = np.random.normal(550, 45) - (age - 65) * 2.2
        else:          # Alzheimer's Disease
            hippo = np.random.normal(2450, 450) - (age - 65) * 22
            ventricle = np.random.normal(50200, 18000) + (age - 65) * 500
            gm = np.random.normal(480, 50) - (age - 65) * 3.5

        hippo_vol[idx] = max(1200, hippo)
        ventricle_vol[idx] = max(10000, ventricle)
        gray_matter_vol[idx] = max(300, gm)

    df['mri_hippocampal_volume'] = np.round(hippo_vol, 1)
    df['mri_ventricle_volume'] = np.round(ventricle_vol, 1)
    df['mri_gray_matter_volume'] = np.round(gray_matter_vol, 1)

    df.to_csv(output_path, index=False)
    print(f"[+] Complete structural dataset successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_stage3_numerics()