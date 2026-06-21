import pandas as pd
import numpy as np

data = pd.read_csv("./data/stage1_synthetic_data.csv")

print(data.head(100))
print(data.describe())