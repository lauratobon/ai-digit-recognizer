
#import needed libraries
import os
os.environ['KAGGLE_CONFIG_DIR'] = "."
import numpy as np
import pickle
import pandas as pd
from keras.models import load_model

print ("running predict")


# Load the pre saved model
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)
model = load_model('my_model.h5')
test = pd.read_csv('test.csv')
x_test = test

#Normalize
x_test  = x_test/255.

#Reshape images
x_test = x_test.values.reshape(-1,28,28,1)

#Use the model to make a prediction
results = model.predict(x_test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

print(results)

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)

submission = pd.read_csv('submission.csv')
print(submission.head())
print(submission.info())