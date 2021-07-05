FROM continuumio/miniconda3:latest

RUN apt update && apt -y upgrade
RUN apt update && apt install -y build-essential wget doxygen libboost-all-dev
RUN conda install -y mkl mkl-include pybind11 numpy sphinx pytest scikit-learn h5py psutil pyyaml pandas
RUN conda install -y -c conda-forge faiss-cpu

RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.0-rc2/cmake-3.21.0-rc2-linux-x86_64.sh
RUN bash cmake-3.21.0-rc2-linux-x86_64.sh --skip-license --prefix=/usr/

ENV MKL_ROOT=/opt/conda

RUN git clone https://github.com/mkarppa/deann deann-source && mkdir -p deann-source/build && cd deann-source/build && cmake .. && make && make install

RUN git clone https://github.com/maumueller/rehashing && cd rehashing/hbe && \
    git submodule init && git submodule update && \ 
    mkdir build && cd build && cmake ../lib/eigen-git-mirror && cd ../lib/config4cpp && make && cd ../.. && \
    cmake . && make

ENV PATH="/rehashing/hbe/:${PATH}"

WORKDIR /app/



