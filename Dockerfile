#FROM python:2.7-slim
FROM continuumio/miniconda

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y g++ libz-dev cmake scons libgsl0-dev libncurses5-dev libxml2-dev libxslt1-dev

WORKDIR /bcr-phylo-benchmark
COPY . /bcr-phylo-benchmark
#RUN ./INSTALL_docker

RUN conda create -y -n bpb
RUN source activate bpb
RUN mkdir -p $CONDA_PREFIX/etc/conda/activate.d
RUN mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
RUN printf '#!/bin/sh\n\nexport PYTHONNOUSERSITE=1' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
RUN printf '#!/bin/sh\n\nunset PYTHONNOUSERSITE' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
RUN source activate bpb
#RUN conda update -y -c bioconda pysam
#RUN conda install -y pyyaml

conda install -c anaconda gxx_linux-64
conda install -y biopython cmake gsl openblas pandas psutil pysam scons seaborn



