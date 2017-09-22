Setup Environment
==================
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.5.0-Linux-x86_64.sh
bash Anaconda*.sh
conda update conda
conda create -n reflux_analyzer python=2.7 anaconda
source activate reflux_analyzer

conda install -n reflux_analyzer django
conda install -n reflux_analyzer keras
pip install opencv-python==3.3.0.10
conda install -n reflux_analyzer -c menpo opencv
conda install -n reflux_analyzer -c conda-forge opencv
conda install -n reflux_analyzer -c menpo openblas
conda install -n reflux_analyzer -c conda-forge djangorestframework

sudo apt-get update
sudo apt-get install python-dev
sudo apt-get update
sudo apt-get upgrade gcc
sudo apt-get install python2.7-dev
sudo apt-get install python-numpy
sudo apt-get install python-pip
sudo pip install --upgrade pip
sudo pip install virtualenv
sudo apt-get install python3-tk
apt-get install python-tk
pip install python-opencv

virtualenv .
source bin/activate
pip install -r requirment.txt
python manage.py collectstatic

