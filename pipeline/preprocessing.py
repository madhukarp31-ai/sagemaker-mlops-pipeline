import pandas as pd
import os

input_path = "/opt/ml/processing/input/adult_final.csv"
train_path = "/opt/ml/processing/output/train/train.csv"
test_path  = "/opt/ml/processing/output/test/test.csv"

df = pd.read_csv(input_path)
print(f"✅ Loaded shape: {df.shape}")

train = df.sample(frac=0.8, random_state=42)
test  = df.drop(train.index)

os.makedirs("/opt/ml/processing/output/train", exist_ok=True)
os.makedirs("/opt/ml/processing/output/test",  exist_ok=True)

train.to_csv(train_path, index=False)
test.to_csv(test_path,   index=False)

print(f"✅ Train : {train.shape}")
print(f"✅ Test  : {test.shape}")
print("✅ Preprocessing complete!")