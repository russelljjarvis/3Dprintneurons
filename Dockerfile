#author Russell Jarvis rjjarvis@asu.edu
#author Rick Gerkin rgerkin@asu.edu

FROM ubuntu
RUN mkdir /home/jovyan/
ENV HOME /home/jovyan/

WORKDIR $HOME
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

RUN wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
ENV PATH "$HOME/miniconda/bin:$PATH"
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
RUN ./configure --prefix=`pwd` --without-iv --with-nrnpython=/home/jovyan/miniconda/bin/python --with-paranrn=/usr/local/lib/openmpi
RUN make all
RUN sudo make install

# Create python bindings for NEURON
WORKDIR src/nrnpython
RUN python setup.py install
ENV NEURON_HOME $HOME/nrn-7.4/x86_64
ENV PATH $NEURON_HOME/bin:$PATH


RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y apt-utils software-properties-common

#RUN echo 'dont use cache'
#RUN git clone https://github.com/makerbot/pyserial.git
#WORKDIR makerbot
#RUN git submodule update --init
#RUN python virtualenv.py virtualenv
#RUN ./setup.sh
#RUN python setup.py install
#RUN git submodule update --init
#WORKDIR submodule/conveyor_bins
#RUN easy_install pyserial-2.7_mb2.1-py2.7.egg
RUN pip install pyserial

RUN apt-add-repository 'http://downloads.makerbot.com/makerware/ubuntu' -y
RUN apt-add-repository ppa:fkrull/deadsnakes -y
RUN wget http://downloads.makerbot.com/makerware/ubuntu/dev@makerbot.com.gpg.key
RUN apt-key add dev@makerbot.com.gpg.key
#RUN apt-get install -y povray# apt-utils# --reinstall
RUN apt-get update
WORKDIR $HOME
#RUN apt-get download makerware #--allow-unauthenticated
#RUN dpkg -i makerware_3.7.0-12.04_amd64.deb
#RUN apt-get -f install
#RUN dpkg -i makerware_3.7.0-12.04_amd64.deb

#RUN apt-get install -y aptitude
#RUN aptitude install -y makerware
#RUN apt-get install -y -f makerware


RUN ln -s /etc/povray/3.7/povray.conf $HOME/.povray
RUN conda install vtk


RUN apt-get update \
      && apt-get install -y sudo \
      && rm -rf /var/lib/apt/lists/*
RUN echo "jovyan ALL=NOPASSWD: ALL" >> /etc/sudoers

RUN useradd -ms /bin/bash jovyan


RUN apt-get update \
      && apt-get install -y sudo \
      && rm -rf /var/lib/apt/lists/*
RUN echo "jovyan ALL=NOPASSWD: ALL" >> /etc/sudoers

RUN chown -R jovyan $HOME

RUN wget http://www.vtk.org/files/release/7.1/vtkpython-7.1.0-Linux-64bit.tar.gz
RUN pip install mayavi
RUN conda install mayavi

USER jovyan
RUN sudo chown -R jovyan $HOME
RUN mpiexec -np 4 python -c "import neuron"
#The following are convenience aliases
RUN echo 'alias pynml=pynml-povray NML2_MultiCompCellNetwork.nml -scalez 15 -conns -conn_points'
RUN echo 'alias nml2= povray NML2_MultiCompCellNetwork.pov'

RUN echo 'alias nb="jupyter notebook --browswer iceweasel"' >> ~/.bashrc
RUN echo 'alias st="cd /home/mnt/neuronunit/software_tests"' >> ~/.bashrc
RUN echo 'alias mnt="cd /home/mnt"' >> ~/.bashrc
RUN echo 'alias erc="emacs ~/.bashrc"' >> ~/.bashrc
RUN echo 'alias src="source ~/.bashrc"' >> ~/.bashrc
RUN echo 'alias egg="cd /opt/conda/lib/python3.5/site-packages/"' >> ~/.bashrc
RUN echo 'alias nu="cd /home/jovyan/work/scidash/neuronunit"' >> ~/.bashrc
RUN echo 'alias model="cd /work/scidash/neuronunit/neuronunit/models"' >> ~/.bashrc
RUN echo 'alias sciunit="cd /work/scidash/sciunit"' >> ~/.bashrc
RUN echo 'alias nu="python -c "from neuronunit.models.reduced import ReducedModel""'
RUN export PATH=$PATH


RUN wget http://www.patrickmin.com/meshconv/download.php?id=4
RUN chown -R jovyan meshconv
RUN chmod -x meshconv

#RUN ls *.wrl >> convert_list.sh
#$HOME/Downloads/./meshconv *.wrl -c stl

#ENV ATOM_VERSION v1.12.7
#RUN sudo apt-get update
#RUN sudo apt-get install -y gedit meld gvfs-bin iceweasel
#RUN sudo  apt-get install -y curl apt-utils



#RUN sudo curl -L https://github.com/atom/atom/releases/download/${ATOM_VERSION}/atom-amd64.deb > /tmp/atom.deb && dpkg -i /tmp/atom.deb && rm -f /tmp/atom.deb
