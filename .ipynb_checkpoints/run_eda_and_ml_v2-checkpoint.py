import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")

# -----------------------------------------
# Loop over v1 and v2
# -----------------------------------------
versions = ['v1', 'v2']
results = []

for DATASET_VERSION in versions:
    print(f"\n🔁 Running analysis for {DATASET_VERSION.upper()}...")

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------
    filename = f"athletes_{DATASET_VERSION}.csv"
    print(f"📂 Loading: {filename}")
    df = pd.read_csv(filename)

    # Ensure total_lift exists
    if 'total_lift' not in df.columns:
        df['total_lift'] = df[['deadlift', 'candj', 'snatch', 'backsq']].sum(axis=1)

    # -----------------------------------------
    # EDA: Save plots to file
    # -----------------------------------------
    print("🧠 Running EDA...")

    # Distribution of total_lift
    plt.figure(figsize=(8, 5))
    sns.histplot(df['total_lift'], bins=30, kde=True)
    plt.title('Distribution of Total Lift')
    plt.xlabel('Total Lift')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"eda_total_lift_dist_{DATASET_VERSION}.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"eda_corr_heatmap_{DATASET_VERSION}.png")
    plt.close()

    # Boxplot by gender
    if 'gender' in df.columns:
        plt.figure(figsize=(7, 5))
        sns.boxplot(x='gender', y='total_lift', data=df)
        plt.title('Total Lift by Gender')
        plt.tight_layout()
        plt.savefig(f"eda_boxplot_gender_{DATASET_VERSION}.png")
        plt.close()

    # Count plot by region
    if 'region' in df.columns:
        plt.figure(figsize=(10, 4))
        sns.countplot(y='region', data=df, order=df['region'].value_counts().index)
        plt.title('Athlete Count by Region')
        plt.tight_layout()
        plt.savefig(f"eda_region_count_{DATASET_VERSION}.png")
        plt.close()

    # -----------------------------------------
    # Train/Test Split
    # -----------------------------------------
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Model features
    features = ['deadlift', 'candj', 'snatch', 'backsq']
    target = 'total_lift'

    # Drop rows with missing values in features/target
    train_df = train_df.dropna(subset=features + [target])
    test_df = test_df.dropna(subset=features + [target])

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # -----------------------------------------
    # Train model
    # -----------------------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Baseline Model Metrics ({DATASET_VERSION}):")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    results.append({
        'version': DATASET_VERSION,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 2)
    })

    # -----------------------------------------
    # Save Actual vs Predicted Plot
    # -----------------------------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"Actual vs Predicted Total Lift ({DATASET_VERSION})")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"features/pred_vs_actual_{DATASET_VERSION}.png")
    plt.close()

# -----------------------------------------
# Save Metrics to CSV
# -----------------------------------------
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("baseline_model_metrics.csv", index=False)
print("\n✅ Saved all outputs and metrics to file.")

# -----------------------------------------
# Generate Markdown Summary
# -----------------------------------------
md_lines = []

md_lines.append("# 🧠 Baseline Model Comparison: v1 vs v2\n")
md_lines.append("| Version | MAE | RMSE | R² Score |")
md_lines.append("|---------|-----|------|----------|")

for row in results:
    md_lines.append(f"| {row['version']} | {row['MAE']} | {row['RMSE']} | {row['R2']} |")

import os

# Find all result figures (sorted by name for consistency)
fig_files = sorted([
    f for f in os.listdir() if f.endswith('.png') and (
        f.startswith('pred_vs_actual_') or f.startswith('eda_')
    )
])

# Append figure markdown
md_lines.append("\n### 📊 Visualizations:\n")
for fig in fig_files:
    md_lines.append(f"#### {fig.replace('_', ' ').replace('.png', '').title()}")
    md_lines.append(f"![{fig}](./{fig})\n")

# Save full markdown with figures
with open("baseline_model_summary.md", "w") as f:
    f.write("\n".join(md_lines))

print("📄 Updated Markdown summary with embedded figures.")

# Save to markdown file
with open("baseline_model_summary.md", "w") as f:
    f.write("\n".join(md_lines))

print("📄 Saved Markdown summary: baseline_model_summary.md")
