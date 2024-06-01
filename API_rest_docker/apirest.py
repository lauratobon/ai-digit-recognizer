#import needed libraries
import os
os.environ['KAGGLE_CONFIG_DIR'] = "."
import numpy as np
import pickle
import pandas as pd
from keras.models import load_model
from flask import Flask, render_template, request, jsonify, session
import matplotlib.pyplot as plt

import seaborn as sns
import tensorflow as tf
import joblib
from werkzeug.utils import secure_filename
from distutils.log import debug
from fileinput import filename

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route('/', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':

        # upload file flask
        f = request.files.get('file')

        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)

        f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))

        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],data_filename)

        return render_template('index2.html')
    return render_template("index.html")

@app.route('/show_data')
def showData():
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path,encoding='unicode_escape')
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html', data_var=uploaded_df_html)

@app.route('/test', methods=['GET'])
def hello():
    return jsonify({'message': 'Server is running!'})

@app.route('/predict_table', methods=['GET'])
def predictTable():

    print ("running predict")

    try:
        # data = request.get_json()
        model = load_model('my_model.h5')

        test = pd.read_csv('test_number.csv')
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
        submission = submission.to_json()
        return jsonify({'message': 'Predicted!', 'results': submission})

        
    except Exception as e:
        return jsonify({'message': 'Error! ' + str(e)})
    

    
   
@app.route('/predict', methods=['GET'])
def predict():

    print ("running predict")

    try:

        model = load_model('my_model.h5')
        # test = pd.read_csv('test_number.csv')
        # x_test = test

        data_file_path = session.get('uploaded_data_file_path', None)
        # read csv
        uploaded_df = pd.read_csv(data_file_path,encoding='unicode_escape')
        # test = pd.read_csv('test.csv')
        x_test = uploaded_df
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
        submission = pd.concat([pd.Series(range(1,len(results)+1),name = "ImageId"),results],axis = 1)
        submission = submission.to_json()
        return jsonify({'message': 'Predicted!, the first number is:' + str(results[0]), 'results': submission})

        
    except Exception as e:
        return jsonify({'message': 'Error! ' + str(e)})
    


@app.route('/train', methods=['GET'])
def train():
    try:
        test = pd.read_csv('test.csv')
        train = pd.read_csv('train.csv')
        train.head()

        print("train shape: ",train.shape)
        print("test shape: ",test.shape)
        print(train['label'].unique())
        print(train['label'].nunique())

        test.head()
        train.head()
        y_train = train.iloc[:,:1]
        x_train = train.iloc[:,1:]
        x_test = test
        print("x_train shape: ", x_train.shape)
        print("y_train shape: ", y_train.shape)
        print("x_test shape: ", x_test.shape)

        #Normalize data

        x_train = x_train/255.
        x_test  = x_test/255.

        #Reshape images

        x_train = x_train.values.reshape(-1,28,28,1)
        x_test = x_test.values.reshape(-1,28,28,1)

        #One-hot encoding

        num_classes=10
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)

        print("x_train shape: ", x_train.shape)
        print("y_train shape: ", y_train.shape)
        print("x_test shape: ", x_test.shape)

        print(x_train.ndim)

        #Printing original labels of top 5 rows
        print(train['label'].head())

        #One hot encoding of the same labels
        print(y_train[0:5,:])

        fig = plt.figure(figsize = (11, 12))

        for i in range(16):
            plt.subplot(4,4,1 + i)
            plt.title(np.argmax(y_train[i]),fontname="Aptos",fontweight="bold")
            plt.imshow(x_train[i,:,:,0], cmap=plt.get_cmap('gray'))
        plt.show()  

        #Defining the model
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation = 'relu', input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
        tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
        tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'Same'),
        tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        #tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.summary()

        #Defining the callback function to stop our training once the acceptable accuracy is reached
        class myCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs={}):
                    if(logs.get('accuracy') > 0.7):
                        print("\nReached 90% accuracy so cancelling training!")
                        self.model.stop_training = True

        callbacks = myCallback()

        #Compiling and model training with batch size = 256, epochs = 100, and optimizer = adam
        Optimizer = tf.keras.optimizers.Adam(
                    learning_rate=0.0005,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07,
                    name='Adam'
        )
        model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size = 256, epochs = 100, callbacks=[callbacks])

        # Model Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')

        # Model Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')

        # Show the graphic
        plt.tight_layout()
        plt.show()

        #Saving the model to be used in the predict file

        model.save('my_model.h5', "./")  # creates a HDF5 file 'my_model.h5'

        # with open('model.pkl', 'wb') as f:
        #      pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        return jsonify({'message': 'Error! ' + str(e)})
    
    return jsonify({'message': 'Trained!'})

if __name__ == '__main__':
    app.run(debug=True)