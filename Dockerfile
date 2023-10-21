FROM registry.linuxone.cloud.marist.edu/jupyterlab-image-s390x:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
            apt-utils && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
        apt-utils \
        bc \
        bzip2 \
        ca-certificates \
        curl \
        git \
        libgdal-dev \
        libssl-dev \
        libffi-dev \
        libncurses-dev \
        libgl1 \
        jq \
        nfs-common \
        parallel \
        python-dev \
        python-pip \
        python-wheel \
        python-setuptools \
        unzip \
        vim \
        tmux \
        wget \
        build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python
&& pip install numpy \
&& pip install pandas \
&& pip install tqdm \
&& pip install seaborn
&& pip install matplotlib \
&& pip install albumentations \
&& pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
&& pip install -q -U segmentation-models-pytorch

RUN ["/bin/bash"]