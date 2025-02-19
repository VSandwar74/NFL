import pandas as pd
import numpy as np

# Load the original CSV
df = pd.read_csv("tracking_week_9.csv")

# Filter rows where frameType is "SNAP"
df_snap = df[df["frameType"] == "SNAP"]

# Shuffle the filtered rows
df_snap = df_snap.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into 9 smaller DataFrames
chunks = np.array_split(df_snap, 9)

# Save each chunk as a separate CSV
for i, chunk in enumerate(chunks):
    chunk.to_csv(f"output_part_{i+1}.csv", index=False)

print("Filtered CSV successfully split into 9 parts!")
