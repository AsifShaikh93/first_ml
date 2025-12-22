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
    pydantic \
    joblib \
    fastapi \
    jinja2 \
    uvicorn[standard] \
    scikit-learn \
    mlflow \
    joblib \
    kfp==2.7.0

# Important: make mlops importable
ENV PYTHONPATH=/opt/mlops
