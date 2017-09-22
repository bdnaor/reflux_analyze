Setup Environment
==================
sudo apt-get update
sudo apt-get upgrade gcc
sudo apt-get install python2.7-dev
sudo apt-get install python-numpy
sudo apt-get install python-pip
sudo pip install virtualenv
sudo apt-get install python3-tk
apt-get install python-tk
sudo apt-get install libopencv-dev python-opencv
virtualenv .
source bin/activate
pip install -r requirment.txt
python manage.py collectstatic

