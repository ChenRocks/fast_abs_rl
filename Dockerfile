FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

#set up environment
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip libxml-parser-perl
RUN pip3 install --upgrade pip

RUN mkdir src
WORKDIR src/
ADD requirements.txt .
RUN pip3 install torch==0.4.0 -f https://download.pytorch.org/whl/cu90/stable
RUN pip3 install -r requirements.txt

ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=latin-1