FROM nvcr.io/nvidia/pytorch:19.01-py3

ENV WORKSPACE /git
WORKDIR $WORKSPACE

ENV DEBIAN_FRONTEND=noninteractive

# Install training code dependencies
RUN pip install --upgrade pip && \
    pip install tensorboardx && \
    pip install pynvml && \
    pip install torch-encoding
