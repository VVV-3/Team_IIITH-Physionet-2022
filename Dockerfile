FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER vijay.vignesh@research.iiit.ac.in

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install -r requirements.txt
RUN conda install -c conda-forge librosa
