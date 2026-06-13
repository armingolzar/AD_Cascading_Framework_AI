import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_stage1_dataset(num_patients=2500, seed=42, output_dir="data"):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Base Demographics (Independent Variables)
    # Education years: Mean 14 years, standard deviation 3
    education_years = np.clip(np.random.normal(14.2, 2.8, size=num_patients), 6, 20)
    biological_sex = np.random.choice([0, 1], size=num_patients, p=[0.47, 0.53]) # 0: Male, 1: Female
    
    # 2. Determine True Latent Clinical Diagnosis Class based on Age
    # We generate age first, then use age to skew the true diagnostic class probability
    base_age = np.random.normal(72.0, 6.5, size=num_patients)
    diagnostic_classes = []
    final_ages = []
    
    for age in base_age:
        # Age-dependent probability shift
        if age < 65:
            p_cn, p_mci, p_ad = 0.85, 0.13, 0.02
            age_adj = age + np.random.normal(0, 1.5)
        elif age < 75:
            p_cn, p_mci, p_ad = 0.50, 0.38, 0.12
            age_adj = age + np.random.normal(0, 1.0)
        else:
            p_cn, p_mci, p_ad = 0.20, 0.45, 0.35
            age_adj = age + np.random.normal(0, 0.5)
            
        cls = np.random.choice([0, 1, 2], p=[p_cn, p_mci, p_ad])
        diagnostic_classes.append(cls)
        final_ages.append(max(50, int(age_adj))) # Floor age at 50
        
    diagnostic_classes = np.array(diagnostic_classes)
    final_ages = np.array(final_ages)

    mmse_scores = np.zeros(num_patients, dtype=int)
    moca_scores = np.zeros(num_patients, dtype=int)
    faq_scores = np.zeros(num_patients, dtype=float) # Functional Activities Questionnaire
    cdr_sb = np.zeros(num_patients, dtype=float)

    for i in range(num_patients):
        cls = diagnostic_classes[i]
        edu = education_years[i]

        # Apply Cognitive Reserve Modifier: High education masks early cognitive drops
        reserve_modifier = 0 if edu < 12 else (1 if edu < 16 else 2)

        if cls == 0: # Cognitively Normal
            mmse_base = np.random.normal(29.1, 0.9)
            moca_base = np.random.normal(27.4, 1.2)
            faq_base = np.random.exponential(scale=0.3)
            cdr_base = np.random.choice([0.0, 0.5], p=[0.95, 0.05])
        elif cls == 1: # Mild Cognitive Impairment (MCI)
            # MoCA drops faster and is more sensitive to MCI than MMSE
            mmse_base = np.random.normal(26.2, 1.5) + (reserve_modifier * 0.5)
            moca_base = np.random.normal(21.8, 2.1) + (reserve_modifier * 0.7)
            faq_base = np.random.normal(2.1, 1.8)
            cdr_base = np.random.choice([0.5, 1.0, 2.0], p=[0.75, 0.20, 0.05])
        else: # Alzheimer's Disease (AD)
            mmse_base = np.random.normal(18.5, 3.8) + (reserve_modifier * 0.3)
            moca_base = np.random.normal(12.4, 4.2) + (reserve_modifier * 0.4)
            faq_base = np.random.normal(16.4, 5.5)
            cdr_base = np.random.normal(6.5, 2.5)

        mmse_scores[i] = int(np.clip(mmse_base, 0, 30))
        moca_scores[i] = int(np.clip(moca_base, 0, 30))
        faq_scores[i] = np.round(np.clip(faq_base, 0, 30), 1)
        cdr_sb[i] = np.round(np.clip(cdr_base, 0, 18), 1)

    df = pd.DataFrame({
        'patient_id': [f"SIM_AD_{idx:05d}" for idx in range(num_patients)],
        'age': final_ages,
        'sex': biological_sex,
        'education_years': np.round(education_years, 1),
        'mmse': mmse_scores,
        'moca': moca_scores,
        'faq': faq_scores,
        'cdr_sb': cdr_sb,
        'diagnostic_class': diagnostic_classes # Target Matrix
    })

    df.to_csv(os.path.join(output_dir, "stage1_synthetic_data.csv"), index=False)

    print(f"Data Generation Complete. Data saved to '{output_dir}/'")
    print(f"Class Distribution: {np.bincount(diagnostic_classes)}")

if __name__ == "__main__":
    generate_stage1_dataset()

