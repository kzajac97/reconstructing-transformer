FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY requirements.txt .

RUN pip install -r -q requirements.txt

COPY notebooks/ ./workspace
COPY src/ ./workspace

ENV WANDB_API_KEY: $WANDB_API_KEY  # add to W&B key to env variables when running docker build
