FROM pytorch/pytorch

COPY requirements.txt .

RUN pip install -r requirements.txt -q

# the directories are copied into workspace directory
COPY notebooks/ .
COPY src/ .
