import numpy as np 
import pandas as pd 
import os

def generate_stage2_biomarkers(stage1_path, output_path, seed=42):

    np.random.seed(seed)

    if not os.path.exists(stage1_path):
        raise FileNotFoundError(f"Could not find Stage 1 data at {stage1_path}. Please check your path.")

    df = pd.read_csv(stage1_path)
    num_patients = len(df)

    print(f"Loaded Stage 1 dataset with {num_patients} patients.")
    print("Generating conditionally mapped Stage 2 FLUID & BLOOD biomarkers...")

    categories = df['diagnostic_class'].values
    mmse = df["mmse"].values
    age = df["age"].values


    csf_amyloid = np.zeros(num_patients, dtype=np.float32)
    csf_ptau = np.zeros(num_patients, dtype=np.float32)
    plasma_ptau217 = np.zeros(num_patients, dtype=np.float32)
    plasma_nfl = np.zeros(num_patients, dtype=np.float32)
    apoe4_count = np.zeros(num_patients, dtype=np.int64)

    for i in range(num_patients):
        cat = categories[i]
        patient_mmse = mmse[i]
        patient_age = age[i]


        if cat == 0:
            base_amyloid = 1000.0
            amyloid_noise = np.random.normal(0, 120)
        elif cat == 1:
            base_amyloid = 620.0 + (patient_mmse - 24) * 15
            amyloid_noise = np.random.normal(0, 100)
        else:
            base_amyloid = 410.0 + (patient_mmse - 15) * 8
            amyloid_noise = np.random.normal(0, 60)
            
        csf_amyloid[i] = max(200.0, base_amyloid, amyloid_noise)

        ############################################################

        if cat == 0:
            base_ptau = 22.0
            ptau_noise = np.random.normal(0, 5)
        elif cat == 1:
            base_ptau = 48.0 - (patient_mmse - 24) * 1.5
            ptau_noise = np.random.normal(0, 10)
        else:
            base_ptau = 85.0 - (patient_mmse - 15) * 2.0
            ptau_noise = np.random.normal(0, 15)

        csf_ptau[i] = max(5.0, base_ptau + ptau_noise)

        ############################################################

        if cat == 0:
            base_p217 = 0.8
            p217_noise = np.random.normal(0, 0.2)
        elif cat == 1:
            base_p217 = 2.4 - (patient_mmse - 24) * 0.2
            p217_noise = np.random.normal(0, 0.5)
        else:
            base_p217 = 7.5 - (patient_mmse - 15) * 0.4
            p217_noise = np.random.normal(0, 1.2)

        plasma_ptau217[i] = max(0.1, base_p217 + p217_noise)

        ############################################################

        age_affect = (patient_age - 70) * 0.5 

        if cat == 0:
            base_nfl = 18.0 + age_affect
            nfl_noise = np.random.normal(0, 0.3)
        elif cat == 1:
            base_nfl = 34.0 + age_affect - (patient_mmse - 24) * 1.0
            nfl_noise = np.random.normal(0, 6.0)
        else:
            base_nfl = 62.0 + age_affect - (patient_mmse - 15) * 1.8
            nfl_noise = np.random.normal(0, 10.0)

        plasma_nfl[i] = max(5.0, base_nfl + nfl_noise)

        ############################################################

        if cat == 0:
            probs = [0.75, 0.22, 0.03]
        elif cat == 1:
            probs = [0.45, 0.43, 0.12]
        else:
            probs = [0.30, 0.50, 0.20]

        apoe4_count[i] = np.random.choice([0,1,2], p=probs)


    df["csf_amyloid_beta"] = np.round(csf_amyloid, 1)
    df["csf_ptau"] = np.round(csf_ptau, 1)
    df["plasma_ptau217"] = np.round(plasma_ptau217, 1)
    df["plasma_nfl"] = np.round(plasma_nfl, 1)
    df["apoe4_alleles"] = apoe4_count

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Success! Clean fluid/blood Stage 2 dataset securely saved to: {output_path}")

if __name__ == "__main__":
    stage1_data = "./stage1_synthetic_data.csv"
    stage2_output = "./stage2_synthetic_data.csv"
    generate_stage2_biomarkers(stage1_data, stage2_output)





