### Created a Flask app that uses PyTorch
* Created a Flask app that loads PyTorch tensor data and a trained model and makes some predictions.
* The code to load the model and make predictions is in the `test` folder.
* Created the Flask app in `main.py` 

### 2. Setup Google Cloud
* Created a new gcloud project 
* Activated Cloud Run and Cloud Build APIs

### 3. Install gcloud SDK
* https://cloud.google.com/sdk/docs/install

### 4. Create Dockerfile, requirements.txt and .dockerignore
* https://cloud.google.com/run/docs/quickstarts/build-and-deploy#containerizing

### 5. Cloud build && deploy
`gcloud builds submit --tag gcr.io/<project_id>/<function_name>`

`gcloud run deploy --image gcr.io/<project_id>/<function_name> --platform managed`