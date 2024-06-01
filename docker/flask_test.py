from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def hello():
    return jsonify({'message': 'Server is running!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
    except Exception as e:
        return jsonify({'message': 'Error! ' + str(e)})
    return jsonify({'message': 'Predicted!'})
    
   

@app.route('/train', methods=['GET'])
def train():
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({'message': 'Error! ' + str(e)})
    
    return jsonify({'message': 'Trained!'})