import os 
import pandas as pd 
import numpy as np
import torch
from model.stage1_screening import Stage1_Encoder
from model.stage2_biomarkers import Stage2_Encoder

class ClinicalPOMDPEnv():

    """
    The RL Playground. Manages patient routing, applies zero-masking,
    charges diagnostic costs, and runs your pre-trained each Stage models.
    """

    def __init__(self, data_path=None, stage1_model_path=None, stage2_model_path=None):
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

        # Total observation space: 7 (Stage 1) + 3 (Stage 1 model predictions) + 5 (Stage 2) + 3 + 3 (Stage 3) = 21 elements
        self.obs_dim = len(self.stage1_cols) + 3 + len(self.stage2_cols) + 3 + len(self.stage3_cols)


        # 3. Clinical Economy
        self.reward_correct = 10.0
        self.penalty_incorrect = -15.0
        self.cost_stage2_blood = -1.5
        self.cost_stage3_mri = -8.0
        self.invalid_action_penalty = -20


        # 4. Device Management and Model Placeholders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage1_model = self._load_model(stage1_model_path, "Stage_1")
        self.stage2_model = self._load_model(stage2_model_path, "Stage_2")


        # Volatile step state trackers
        self.current_patient_idx = 0
        self.current_stage = 1
        self.patient_row = None
        self.true_class = None

        # Tracking independent model probability vectors
        self.stage1_probs = np.zeros(3, dtype=np.float32)
        self.stage2_probs = np.zeros(3, dtype=np.float32)


    # def _load_model(self, model_path, name):
    #     """Helper to safely alert if a model file path is ready to load."""
    #     if model_path is not None and os.path.exists(str(model_path)):
    #         try:
    #             if 




