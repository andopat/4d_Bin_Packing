# CPU based docker
FROM ubuntu:16.04

## Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cmake \
        git \
        wget \
        nano \
        vim \
        cdo \
        ncl-ncarg \
        software-properties-common \
        python-software-properties
        

# PYTHON 3.6
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv


# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel

COPY ./requirements.txt /work/requirements.txt

WORKDIR /work

RUN python3.6 -m pip install -r requirements.txt


# copy entire directory where docker file is into docker container at /work
COPY . /work/