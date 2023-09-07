import pandas as pd
import numpy as np
from numpy import array
import tensorflow as tf
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow import keras
import tensorflowjs as tfjs


df = pd.read_csv('data2.csv')
unique_soil_types = df['Soil_Type'].unique()
unique_crop_types = df['Crop_Type'].unique()
print(unique_soil_types)
print(unique_crop_types)
num_soil_types = len(unique_soil_types)
num_crop_types = len(unique_crop_types)

print(f"Number of unique Soil Types: {num_soil_types}")
print(f"Number of unique Crop Types: {num_crop_types}")
soil_type= array(df['Soil_Type'].tolist())
crop_type= array(df['Crop_Type'].tolist())
label_encoder = LabelEncoder()
integer_encoded1 = label_encoder.fit_transform(soil_type)
integer_encoded2 = label_encoder.fit_transform(crop_type)
dict1={}
for i in range(len(integer_encoded1)):
  dict1[soil_type[i]]=integer_encoded1[i]
dict2={}
for i in range(len(integer_encoded2)):
  dict2[crop_type[i]]=integer_encoded2[i]

df['Soil_Type']=df['Soil_Type'].map(dict1)
df['Crop_Type']=df['Crop_Type'].map(dict2)
print(df['Fertilizer_Name'].unique())
print(df['Fertilizer_Name'].value_counts())
features=df
target = features.pop('Fertilizer_Name')
label = target
labels_onehot = pd.get_dummies(label)
print(label)

X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.2, random_state=2)


model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(8,), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(labels_onehot.shape[1], activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2000, batch_size=32, validation_data=(X_test, y_test))
#INCREASE EPOCHS PLEASE TY 

tfjs.converters.save_keras_model(model, 'tfjs_model')

