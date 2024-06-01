FROM juanarismendy/scikit_modified

RUN pip install -U Flask loguru numpy
RUN mkdir /app

COPY apirest.py /app

COPY model.pkl /app
COPY my_model.h5 /app
COPY kaggle.json /app   
COPY train.csv /app
COPY test.csv /app
COPY test_number.csv /app
COPY templates /app/templates

WORKDIR /app
RUN mkdir -p staticFiles/uploads

CMD flask --app apirest run --host=0.0.0.0 --port=5000


