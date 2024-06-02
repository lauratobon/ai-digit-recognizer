# This line defines the base image for the Docker container.
# In this case, it uses the image named "juanarismendy/scikit_modified".
# This image contains a modified version of the scikit-learn library.
FROM juanarismendy/scikit_modified

# Creates a directory named "app" inside the container.
# This directory will be used as the working directory for the application.
RUN mkdir /app

# Copies the Python script "train.py" to the "/app" directory.
# This script is  responsible for training the machine learning model.
COPY train.py /app

# Copies the Python script "predict.py" to the "/app" directory.
# This script is responsible for making predictions using the trained model.
COPY predict.py /app

# Copies the pickled model file "model.pkl" to the "/app" directory.
# This file contains the trained machine learning model.
COPY model.pkl /app

# Copies the "kaggle.json" file to the "/app" directory.
# This file might contain authentication credentials for Kaggle or other purposes.
COPY kaggle.json /app

# Copies the "train.csv" file to the "/app" directory.
# This file contains the training data for the machine learning model.
COPY train.csv /app

# Copies the "test.csv" file to the "/app" directory.
# This file contains the data to make predictions on.
COPY test.csv /app

# Copies the script "run.sh" to the "/app" directory.
# This script might be a shell script that executes the training and prediction logic.
COPY run.sh /app

# Changes the working directory to "/app" inside the container.
# Subsequent commands will be executed from this directory.
WORKDIR /app

# Makes the "run.sh" script executable for all users inside the container.
# This allows the script to be run from the command line.
RUN chmod a+x run.sh

# Defines the default command to run when the container starts.
# This command executes the "./run.sh" script, starting your training or prediction process.
CMD ["./run.sh"]




