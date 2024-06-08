FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY requirements.txt .

RUN pip install -r requirements.txt -q

# the directories are copied into workspace directory
COPY notebooks/ .
COPY src/ .
