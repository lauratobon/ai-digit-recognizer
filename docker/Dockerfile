FROM juanarismendy/scikit_modified


RUN mkdir /app
COPY train.py /app
COPY predict.py /app
COPY model.pkl /app
COPY kaggle.json /app   
COPY train.csv /app
COPY test.csv /app
COPY run.sh /app
WORKDIR /app
RUN chmod a+x run.sh
CMD ["./run.sh"]



