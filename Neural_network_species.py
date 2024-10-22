#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load libraries
get_ipython().system('pip install tensorflow')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# In[22]:


# Load dataset
df = pd.read_csv('/Users/u20744669/Downloads/Iris.csv')
# Display the first few rows
df.head()  


# In[28]:


#Data Processing
# Encode the species labels
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

# Separate features and target variable
X = df.drop(columns=['Species'])
y = df['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[50]:


# Build the model
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 species
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[52]:


# Train the model and store the training history
history = model.fit(X_train, y_train, epochs=100, batch_size=5, validation_split=0.2, verbose=1)



# In[54]:


#Plot the Training history
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='best')

plt.tight_layout()
plt.show()


# In[56]:


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")


# In[60]:


# Make predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Compare with true labels
print(f"True labels: {y_test.values}")
print(f"Predicted labels: {predicted_classes}")


# In[62]:


#Save the Model
# Save the model to a file
model.save('iris_model.h5')


# In[ ]:




