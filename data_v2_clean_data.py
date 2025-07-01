import pandas as pd
import numpy as np

# Load the raw data
data = pd.read_csv("athletes.csv")

# Remove not relevant columns (fix duplicates in list)
data = data.dropna(subset=[
    'region', 'age', 'weight', 'height', 'howlong', 'gender', 'eat',
    'train', 'background', 'experience', 'schedule', 
    'deadlift', 'candj', 'snatch', 'backsq'
])

data = data.drop(columns=[
    'affiliate', 'team', 'name', 'athlete_id', 'fran', 'helen', 'grace',
    'filthy50', 'fgonebad', 'run400', 'run5k', 'pullups', 'train'
], errors='ignore')  # errors='ignore' in case some cols don't exist

# Remove outliers
data = data[data['weight'] < 1500]
data = data[data['gender'] != '--']
data = data[data['age'] >= 18]
data = data[(data['height'] < 96) & (data['height'] > 48)]

# Handle gender-specific deadlift filtering
data = data[
    ((data['deadlift'] > 0) & (data['deadlift'] <= 1105)) |
    ((data['gender'] == 'Female') & (data['deadlift'] <= 636))
]

data = data[(data['candj'] > 0) & (data['candj'] <= 395)]
data = data[(data['snatch'] > 0) & (data['snatch'] <= 496)]
data = data[(data['backsq'] > 0) & (data['backsq'] <= 1069)]

# Clean survey data
decline_dict = {'Decline to answer|': np.nan}
data = data.replace(decline_dict)
data = data.dropna(subset=['background', 'experience', 'schedule', 'howlong', 'eat'])

# Create total_lift column
data['total_lift'] = data['deadlift'] + data['candj'] + data['snatch'] + data['backsq']

# Save cleaned data as v2
data.to_csv("cleaned_athletes.csv", index=False)

print("✅ Cleaned dataset (v2) saved as 'cleaned_athletes.csv'")
