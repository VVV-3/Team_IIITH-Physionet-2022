FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER vijay.vignesh@research.iiit.ac.in

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
