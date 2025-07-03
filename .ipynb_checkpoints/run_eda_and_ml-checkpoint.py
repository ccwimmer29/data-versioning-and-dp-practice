# Change this to 'v1' or 'v2' to switch dataset version
DATASET_VERSION = 'v2'

# -----------------------------------------
# Setup
# -----------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")

# -----------------------------------------
# Load dataset
# -----------------------------------------
filename = f"athletes_{DATASET_VERSION}.csv"
print(f"\n📂 Loading: {filename}")
df = pd.read_csv(filename)

# Ensure total_lift exists
if 'total_lift' not in df.columns:
    df['total_lift'] = df['deadlift'] + df['candj'] + df['snatch'] + df['backsq']

# -----------------------------------------
# EDA
# -----------------------------------------
print("🔍 First 5 rows:")
print(df.head())

print("\n📊 Summary statistics:")
print(df.describe())

print("\n🧼 Data types and missing values:")
df.info()
print(df.isnull().sum())

# Distribution of total_lift
plt.figure(figsize=(8, 5))
sns.histplot(df['total_lift'], bins=30, kde=True)
plt.title('Distribution of Total Lift')
plt.xlabel('Total Lift')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot of total_lift by gender (if exists)
if 'gender' in df.columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='gender', y='total_lift', data=df)
    plt.title('Total Lift by Gender')
    plt.show()

# Count plot of regions (if exists)
if 'region' in df.columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(y='region', data=df, order=df['region'].value_counts().index)
    plt.title('Athlete Count by Region')
    plt.show()

# -----------------------------------------
# Train/Test Split
# -----------------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# -----------------------------------------
# Model Training and Evaluation
# -----------------------------------------
features = ['age', 'height', 'weight', 'candj', 'snatch', 'backsq', 'deadlift']
target = 'total_lift'

# Drop rows with NaNs in selected columns
train_df = train_df.dropna(subset=features + [target])
test_df = test_df.dropna(subset=features + [target])

# Split into X/y
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Baseline Model Metrics ({DATASET_VERSION}):")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
