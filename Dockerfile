FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt install -y tzdata

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      libsndfile1 \
      ffmpeg \
      libaio-dev \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-py311_24.4.0-0-Linux-x86_64.sh &&  /bin/bash /Miniconda3-py311_24.4.0-0-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-py311_24.4.0-0-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

RUN pip install --upgrade pip
RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117



WORKDIR /usr/src/app
COPY requirements.txt ./
RUN apt-get update && apt-get install -y libsndfile1 wget lsb-release software-properties-common
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN bash llvm.sh 16
ENV PATH "/usr/lib/llvm-16/bin/:$PATH"
RUN pip install --use-deprecated=legacy-resolver --no-cache-dir -r requirements.txt

#RUN git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


#RUN apt-get update -y  && apt-get update && apt-get install -y build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
#RUN git clone --recursive -j8 https://github.com/boostorg/boost.git
#RUN cd boost && ./bootstrap.sh && \
# ./b2 --prefix=$HOME/usr --libdir=$PREFIX/lib64 --layout=tagged link=static,shared threading=multi,single install -j4 && cd ..

#RUN git clone https://github.com/kpu/kenlm && cd kenlm && mkdir build && cd build && cmake .. && make -j 4 && cd ..
#RUN cd kenlm && python setup.py install

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader nonbreaking_prefixes
RUN python -m nltk.downloader perluniprops
