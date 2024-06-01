# ai-digit-recognizer

## How to run the Predict/Train API REST server containerized solution 
(Tests were made using a MacBook with M1 processor (ARM architecture)

Assuming you have docker and it's running
1. clone the repository

Bash
```bash
git clone https://github.com/Juanda16/ai-digit-recognizer.git
```

2. go to cd api_rest_docker folder

Bash
```bash
git cd api_rest_docker  
```

3. Pull the base docker file 

Bash
```bash
docker pull juanarismendy/scikit_modified
```


4. build the container based on the Dockerfile

Bash
```bash
docker build -t api_rest . 
```

5. run the container

Bash
```bash
 docker run -it -p 5001:5000 api_rest    
```
This lates step will start running the server 

6. After the server is app and running the are the following endpoints

    http://127.0.0.1:5001/  -> Made to test the full flow, over there you can upload a CSV file to predict a number , you can see the uploaded file and train the model as well.
    ******
    To test it , you can use the csv file inside api_rest_docker called test_file.csv
    ******

    http://127.0.0.1:5001/predict  -> Made to predict a number, you must use a POST method with a CSV given table

    http://127.0.0.1:5001/predict_table  -> Made to predict numbers with a pre loaded table 

    http://127.0.0.1:5001/train  -> Made to train the model every time you need it with a pre loaded table, Just call it with a GET method


## How to run the containerized solution 
(Tests were made using a MacBook with M1 processor (ARM architecture)


Assuming you have docker and it's running
1. clone the repository

Bash
```bash
git clone https://github.com/Juanda16/ai-digit-recognizer.git
```

2. go to docker folder

Bash
```bash
git cd docker
```

3. Pull the base docker file 

Bash
```bash
docker pull juanarismendy/scikit_modified
```


4. build the container based on the Dockerfile
Bash
```bash
docker build -t digit_recognizer .
```

5. run the container
Bash
```bash
 docker run -it digit_recognizer   
```
This lates step will start running the train.py and the predict.py files 

## How to run  the project in colab

1. Download you own kaggle.json from your kaggel account
2. Upload your own Kagggle.json file on the notebook 1 on colab
3. Run the notebook 01 - generate data  Digit_Recognizer to generate sample train and test data, and see how models are working
4. Upload your own Kagggle.json file on the notebook 2 on colab
5. Run the notebook 02-Predict  Digit_Recognizer run scripts to see the predictions
