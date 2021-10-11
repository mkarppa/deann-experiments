FROM continuumio/miniconda3:latest

RUN apt update && apt -y upgrade
RUN apt update && apt install -y build-essential wget doxygen libboost-all-dev
RUN conda install -y mkl mkl-include pybind11 numpy sphinx pytest scikit-learn h5py psutil pyyaml pandas
RUN conda install -y -c conda-forge faiss-cpu

ENV MKL_ROOT=/opt/conda

WORKDIR /app/

COPY make_groundtruth.py ./

ENTRYPOINT ["python3", "make_groundtruth.py"]



