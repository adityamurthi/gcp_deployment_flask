# Python image to use.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image. Will ignore files/objects we want to ignore
# by putting them in .dockerignore
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies. Contains all the Python modules we'd want to install.
RUN pip install -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
# gunicorn is a production server that is better than the barebones development server Flask provides.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app





