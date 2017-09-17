sudo apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

sudo echo 'export PATH=~/anaconda3/bin/:$PATH' >> ~/.bashrc && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

git config --global user.email "diego.amicabile@gmail.com"
git config --global user.name "Diego Amicabile"
git clone git@github.com:diegoami/DA_kaggle.git