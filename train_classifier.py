# Importing Libraries
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert to numpy arrays
data = np.asarray(data_dict.get('data', []))
labels = np.asarray(data_dict.get('labels', []))

# Check if the data is loaded correctly
if data.size == 0 or labels.size == 0:
    raise ValueError("Loaded data or labels are empty. Please check the data source.")

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
