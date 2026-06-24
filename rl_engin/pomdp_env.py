import os 
import pandas as pd 
import numpy as np
import torch

class ClinicalPOMDPEnv():

    """
    The RL Playground. Manages patient routing, applies zero-masking,
    charges diagnostic costs, and runs your pre-trained each Stage models.
    """

    def __init__(self, data_path=None, stage1_model_path=None):
        # 1. Locate and load your completed dataset
        if data_path is None:
            self.data_path = os.path.join("..", "data", "stage3_synthetic_data.csv")
        else:
            self.data_path = data_path

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[-] Missing dataset file at: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        self.num_patients = len(self.df)

        # 2. Define our clinical feature boundaries
        self.stage1_cols = ['age', 'sex', 'education_years', 'mmse', 'moca', 'faq', 'cdr_sb']
        self.stage2_cols = ['csf_amyloid_beta', 'csf_ptau', 'plasma_ptau217', 'plasma_nfl', 'apoe4_alleles']
        self.stage3_cols = ['mri_hippocampal_volume', 'mri_ventricle_volume', 'mri_gray_matter_volume']

        # Total observation space: 7 (Stage 1) + 3 (Stage 1 model predictions) + 5 (Stage 2) + 3 (Stage 3) = 18 elements



