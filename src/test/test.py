import requests
import pickle
import json

from utils import load_data

path_to_data = '/Users/Adi/ML/gcp_deployment_flask/data/bike_rides_test_sample.pt'
resp = requests.post('http://127.0.0.1:5000/predict',
                     json=json.dumps(pickle.dumps(load_data(path_to_data)).decode('latin-1')))

print(resp.json())