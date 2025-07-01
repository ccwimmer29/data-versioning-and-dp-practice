import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

# -----------------------------
# Load and Prepare Data
# -----------------------------
df = pd.read_csv("athletes_v2.csv")

# Select numerical features only
features = ['age', 'height', 'weight']
target = 'total_lift'

# Drop rows with missing values
df = df.dropna(subset=features + [target])

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# DP Model Configuration
# -----------------------------
learning_rate = 0.15
noise_multiplier = 1.1
l2_norm_clip = 1.0
batch_size = 64
epochs = 15
num_microbatches = batch_size

n = (X_train_scaled.shape[0] // batch_size) * batch_size
X_train_scaled = X_train_scaled[:n]
y_train = y_train[:n]

print("N", n)
print("X_train_scaled", X_train_scaled)
print("y_train", y_train)

# -----------------------------
# Define the Model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(features),)),
    tf.keras.layers.Dense(1)
])

optimizer = tfp.DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=2,
    learning_rate=learning_rate
)

loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

# -----------------------------
# Train the Model
# -----------------------------
print("🚀 Training DP model...")
model.fit(
    X_train_scaled,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    shuffle=True,
    verbose=2
)

# -----------------------------
# Evaluate Model
# -----------------------------
print("📊 Evaluating on test set...")
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"🔐 DP Model Metrics: MAE = {mae:.2f}, MSE = {loss:.2f}")

# -----------------------------
# Compute Privacy Guarantee
# -----------------------------
print("🔒 Computing privacy guarantee...")
n = len(X_train_scaled)
epsilon, _ = compute_dp_sgd_privacy(
    n=n,
    batch_size=batch_size,
    noise_multiplier=noise_multiplier,
    epochs=epochs,
    delta=1e-5
)

print(f"✅ Differential Privacy Guarantee: ε = {epsilon:.2f}, δ = 1e-5")
