#author Russell Jarvis rjjarvis@asu.edu
#author Rick Gerkin rgerkin@asu.edu

FROM continuumio/anaconda 
USER root
RUN apt-get -qq update
RUN apt-get -qq -y install curl

RUN apt-get update

RUN apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git gcc g++ build-essential \ 
    openmpi-bin openmpi-doc libopenmpi-dev \
    emacs \
    libxml2-dev libxslt-dev python-dev sudo

#WORKDIR $HOME
RUN apt-get install -y python-setuptools python-dev build-essential
RUN easy_install pip
#RUN apt-get install -y wget python-pip python-setuptools

RUN wget https://www.open-mpi.org/software/ompi/v2.0/downloads/openmpi-2.0.0.tar.gz
RUN tar -xzf openmpi-2.0.0.tar.gz && rm openmpi-2.0.0.tar.gz
WORKDIR $HOME/openmpi-2.0.0

# Compile openmpi
RUN ./configure
RUN make all
RUN sudo make install
    
#RUN pip install mpi4py ipython

RUN pip install --upgrade pip
# Get python bindings for open mpi

#RUN wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.3.0-Linux-x86_64.sh
#RUN bash Anaconda-2.3.0-Linux-x86_64.sh
RUN conda install mpi4py ipython

RUN apt-get install -y libncurses-dev

#Install NEURON-7.4 with python, with MPI
#An infamous build process,and much of the motivation for this container
RUN wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/nrn-7.4.tar.gz
RUN tar -xzf nrn-7.4.tar.gz && rm nrn-7.4.tar.gz 
WORKDIR nrn-7.4
RUN which python
RUN ./configure --prefix=`pwd` --without-iv --with-nrnpython=/opt/conda/bin/python --with-paranrn=/usr/local/lib/openmpi
RUN make all
RUN sudo make install

# Create python bindings for NEURON
WORKDIR src/nrnpython
RUN python setup.py install
ENV NEURON_HOME $HOME/nrn-7.4/x86_64
ENV PATH $NEURON_HOME/bin:$PATH
WORKDIR $HOME


apt-add-repository 'http://downloads.makerbot.com/makerware/ubuntu'
apt-add-repository ppa:fkrull/deadsnakes
wget http://downloads.makerbot.com/makerware/ubuntu/dev@makerbot.com.gpg.key
apt-key add dev@makerbot.com.gpg.key
apt-get update
apt-get install makerware

ENV HOME /home/jovyan/

RUN apt-get update \
      && apt-get install -y sudo \
      && rm -rf /var/lib/apt/lists/*
RUN echo "jovyan ALL=NOPASSWD: ALL" >> /etc/sudoers


RUN chown -R jovyan $HOME

USER $NB_USER
