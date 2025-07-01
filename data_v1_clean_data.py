import pandas as pd

# Load original v1 dataset
df = pd.read_csv("athletes.csv")

# Create total_lift column
df['total_lift'] = df['deadlift'] + df['candj'] + df['snatch'] + df['backsq']

# Save updated v1 dataset
df.to_csv("athletes_v1.csv", index=False)
