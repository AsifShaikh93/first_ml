FROM python:3.10-slim

WORKDIR /opt/mlops

# Copy only what components need
COPY mlops/ mlops/
COPY requirements.txt .

RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    seaborn \
    numpy \
    joblib \
    scikit-learn \
    mlflow \
    joblib 

# Important: make mlops importable
ENV PYTHONPATH=/opt/mlops
