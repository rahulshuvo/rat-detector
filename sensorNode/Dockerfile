FROM python:3.9-bullseye

WORKDIR /src

RUN apt-get update && apt-get upgrade -y && apt-get autoclean
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY src/. .

RUN pip install -r requirements.txt


CMD [ "python", "main.py"]
