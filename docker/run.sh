#!/bin/bash

exec python3 train.py &
wait
exec python3 predict.py