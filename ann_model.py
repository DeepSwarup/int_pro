import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Read data from CSV file
df = pd.read_csv('../assets/disresdata1.1.csv')
data = df.drop('Domain', axis=1)

# Separate input features and target variables
features = data.iloc[:, :18].values
targets = data.iloc[:, 18:].values

# Normalize the input features using Min-Max scaling
feature_scaler = MinMaxScaler()
normalized_features = feature_scaler.fit_transform(features)

# Normalize the target variables using Min-Max scaling
target_scaler = MinMaxScaler()
normalized_targets = target_scaler.fit_transform(targets)

# Define the number of folds
num_folds = 5

# Initialize the KFold object
kfold = KFold(n_splits=num_folds, shuffle=True)

# Define lists to store the evaluation results
all_losses = []
all_accuracies = []
val_losses = []
val_accuracies = []

# Loop over the folds
fold = 1
for train_indices, test_indices in kfold.split(normalized_features):
    print(f"Fold {fold}")

    # Split the data into training and testing sets
    X_train, X_test = normalized_features[train_indices], normalized_features[test_indices]
    y_train, y_test = normalized_targets[train_indices], normalized_targets[test_indices]

    # Define the neural network model with increased complexity
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(18,)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=100, verbose=0)

    # Evaluate the model on training data
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    all_losses.append(train_loss)
    all_accuracies.append(train_accuracy)

    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(X_test, y_test)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Train Loss: {train_loss}")
    print(f"Train Accuracy: {train_accuracy}")
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

    fold += 1

# Print the average performance metrics across all folds
print(f"Train Loss: {all_losses}")
print(f"Validation Loss: {val_losses}")
print(f"Train Accuracy: {all_accuracies}")
print(f"Validation Accuracy: {val_accuracies}")
print(f"Average Train Loss: {np.mean(all_losses)}")
print(f"Average Validation Loss: {np.mean(val_losses)}")
print(f"Average Train Accuracy: {np.mean(all_accuracies)}")
print(f"Average Validation Accuracy: {np.mean(val_accuracies)}")

# plots>>>

# Define the bar width
bar_width = 0.20

# Create an array of x positions for the bars
x_positions = np.arange(len(val_losses))

# Plot the losses
fig, ax = plt.subplots()
ax.bar(x_positions, all_losses, width=bar_width, color='red', label='Loss')
ax.bar([e + bar_width for e in x_positions], val_losses,width=bar_width, color='grey', label='Validation Loss')

# Add labels and title
ax.set_xlabel('Fold')
ax.set_ylabel('Loss')
ax.set_title('Validation Loss for Each Fold')

# Set the x-axis tick labels
ax.set_xticks(x_positions)
ax.set_xticklabels([f'Fold {i+1}' for i in range(len(val_losses))])
plt.yticks(np.arange(0,0.03, 0.0025))
ax.legend()
plt.grid()
plt.show()

# Plot the losses
fig, ax = plt.subplots()
ax.bar(x_positions, all_accuracies, width=bar_width,color='aqua', label='accuracy')
ax.bar([e + bar_width for e in x_positions], val_accuracy,width=bar_width, color='black', label='Validation accuracy')

# Add labels and title
ax.set_xlabel('Fold')
ax.set_ylabel('Acurracy')
ax.set_title('Validation accuracy for Each Fold')

# Set the x-axis tick labels
ax.set_xticks(x_positions)
ax.set_xticklabels([f'Fold {i+1}' for i in range(len(val_losses))])
plt.yticks(np.arange(0, 1, 0.05))

ax.legend()
plt.grid()
plt.show()



