### Shamelessly stolen from this template from:
# https://fmgdata.kinja.com/using-docker-with-conda-environments-1790901398
FROM continuumio/miniconda



# Set the ENTRYPOINT to use bash
# (this is also where you’d set SHELL,
# if your version of docker supports this)
ENTRYPOINT ["/bin/bash", "-c"]

EXPOSE 5000

# Conda supports delegating to pip to install dependencies
# that aren’t available in anaconda or need to be compiled
# for other reasons. In our case, we need psycopg compiled
# with SSL support. These commands install prereqs necessary
# to build psycopg.
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
 libpq-dev \
 build-essential \
 xvfb \
 vim \
 build-essential \
&& rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y libblas-dev liblapack-dev gfortran




RUN conda config --add channels cswarth
RUN conda config --add channels conda-forge
RUN conda install -y -c etetoolkit ete3 ete_toolchain
RUN conda install -y seqmagick biopython matplotlib nestly scons cmake gsl openblas pandas psutil pysam seaborn scipy jellyfish


# Install perl modules:
RUN cpan PDL
#RUN apt-get install gfortran
RUN cpan install PDL::LinearAlgebra::Trans



# Use the environment.yml to create the conda environment.
ADD environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN ["conda", "env", "create"]

WORKDIR /bcr-phylo-benchmark
COPY . /bcr-phylo-benchmark


# Install IgPhyML (required to update hard-coded paths...):
RUN cd tools/IgPhyML && ./make_phyml_omp && cd ../..





###ADD . /code

# Use bash to source our new environment for setting up
# private dependencies—note that /bin/bash is called in
# exec mode directly
###WORKDIR /code/shared
###RUN ["/bin/bash", "-c", "source activate your-environment && python setup.py develop"]

#WORKDIR /code
###RUN ["/bin/bash", "-c", "source activate your-environment && python setup.py develop"]

# We set ENTRYPOINT, so while we still use exec mode, we don’t
# explicitly call /bin/bash
###CMD ["source activate your-environment && exec python application.py"]













#RUN apt-get update && apt-get upgrade -y
#RUN apt-get install -y g++ libz-dev cmake scons libgsl0-dev libncurses5-dev libxml2-dev libxslt1-dev
#RUN apt-get install -y xvfb




#WORKDIR /bcr-phylo-benchmark
#COPY . /bcr-phylo-benchmark
#RUN ./INSTALL_docker

#RUN conda create -y -n bpb
#RUN source activate bpb
#RUN mkdir -p $CONDA_PREFIX/etc/conda/activate.d
#RUN mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
#RUN printf '#!/bin/sh\n\nexport PYTHONNOUSERSITE=1' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#RUN printf '#!/bin/sh\n\nunset PYTHONNOUSERSITE' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
#RUN source activate bpb

#RUN conda update -y -c bioconda pysam
#RUN conda install -y pyyaml

#RUN conda install -c anaconda gxx_linux-64
#RUN conda install -y biopython cmake gsl openblas pandas psutil pysam scons seaborn



