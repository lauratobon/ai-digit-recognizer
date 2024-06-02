# This line defines the base image for the Docker container.
# In this case, it uses the image named "juanarismendy/scikit_modified".
# This image contains a modified version of the scikit-learn library.
FROM juanarismendy/scikit_modified

# Installs the latest versions of Flask, Loguru, NumPy libraries inside the container.
# Flask is used for building the web API, Loguru for logging, and NumPy for numerical computations. 
RUN pip install -U Flask loguru numpy

# Creates a directory named "app" inside the container.
# This directory will be used as the working directory for the application.
RUN mkdir /app

# Copies the Python script "apirest.py" to the "/app" directory.
COPY apirest.py /app

# Copies the pickled model file "model.pkl" to the "/app" directory.
COPY model.pkl /app

# Copies the HDF5 model file "my_model.h5" to the "/app" directory.
# This file contain a Keras or TensorFlow model used by the API.
COPY my_model.h5 /app

# Copies the "kaggle.json" file to the "/app" directory.
# This file might contain authentication credentials for Kaggle or other purposes.
COPY kaggle.json /app

# Copies the "train.csv" file to the "/app" directory.
COPY train.csv /app

# Copies the "test.csv" file to the "/app" directory.
COPY test.csv /app

# Copies the "test_number.csv" file to the "/app" directory.
COPY test_number.csv /app

# Copies the directory "templates" to the "/app/templates" directory inside the container.
# This directory likely contains HTML templates used by your Flask application for building web pages. 
COPY templates /app/templates

# Changes the working directory to "/app" inside the container.
WORKDIR /app

# Creates a directory named "staticFiles/uploads" inside the container with parent directories created if they don't exist (using -p flag).
# This directory be used to store uploaded files by the API.
RUN mkdir -p staticFiles/uploads

# Defines the default command to run when the container starts.
# This command starts the Flask development server using the provided configuration:
#  - --app apirest: Specifies the application module as "apirest.py".
#  - run: Starts the development server.
#  - --host=0.0.0.0: Makes the server accessible from any machine on the network (not just localhost).
#  - --port=5000: Sets the server port to 5000.
CMD flask --app apirest run --host=0.0.0.0 --port=5000



