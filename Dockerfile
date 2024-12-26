FROM python:3.9.8-slim-bullseye
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN /usr/local/bin/python -m pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./src src
COPY ./yolo11n.pt yolo11n.pt
# CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]
CMD ["/bin/sh"]
