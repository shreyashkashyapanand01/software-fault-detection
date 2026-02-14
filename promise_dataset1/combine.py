import pandas as pd
import os

all_dfs = []

# Loop from 1.csv to 34.csv
for i in range(1, 35):
    file_name = f"{i}.csv"
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        all_dfs.append(df)
    else:
        print(f"{file_name} not found")

# Combine all files
combined_df = pd.concat(all_dfs, ignore_index=True)

# Save final dataset
combined_df.to_csv("PROMISE.csv", index=False)

print("PROMISE.csv created successfully")
