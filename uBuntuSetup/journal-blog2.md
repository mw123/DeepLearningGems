sudo apt install python-pip (pip 8.1.1)
sudo -i
pip install virtualenv (virtualenv-15.1.0)
pip install virtualenvwrapper
exit
export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
source /usr/local/bin/virtualenvwrapper.sh
echo 'source /usr/local/bin/virtualenvwrapper.sh' >> ~/.bashrc
mkvirtualenv keras_tf

Note: if you're using a version control system like git, you shouldn't commit the Envs directory. Add it to your .gitignore file (or similar).

pip install --upgrade tensorflow
pip install numpy scipy
pip install scikit-learn
pip install pillow
pip install h5py
pip install keras
  python
  import keras
check .kera/keras.json to make sure backend is set to tensorflow
pip install matplotlib
pip install nose
pip install scikit-image
pip install pandas
pip install sympy

mkdir ~/git
cd ~/git
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make FC=gfortran -j16
sudo make PREFIX=/usr/local install

sudo apt-get update
sudo apt-get install swig (connect c/c++ with high level languages)

OPENCV RELATED:

image format reading:
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev

sudo apt install qtbase5-dev

video format libs:
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev

sudo apt-get install libgtk2.0-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libhdf5-serial-dev

cd ~/git
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
unzip opencv.zip
unzip opencv_contrib.zip

cd opencv-3.1.0

vim ../modules/cudalegacy/src/graphcuts.cpp
+// GraphCut has been removed in NPP 8.0
+#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER) || (CUDART_VERSION >= 8000)

vim ../../opencv_contrib-3.1.0/modules/hdf/include/opencv2/hdf/hdf5.hpp
-#include <hdf5.h>
-using namespace std;
+using namespace std; //in line 45 after namespace hdf {

vim ../../opencv_contrib-3.1.0/modules/hdf/src/hdf5.cpp
+#include <hdf5.h> //in line 37

vim /$HOME/.virtualenv/(virtualenv)/local/lib/python2.7/site-packages/numpy/core/include/numpy/npy_common.h
-#if NPY_INTERNAL_BUILD
+#ifndef NPY_INTERNAL_BUILD
+#define NPY_INTERNAL_BUILD

mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=ON -D WITH_V4L=ON -D WITH_QT=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.1.0/modules -D BUILD_EXAMPLES=ON -D WITH_OPENGL=ON -D WITH_VTK=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..



make -j16
sudo make install
sudo ldconfig

sudo apt-get clean && sudo apt-get autoremove
sudo rm -rf /var/lib/apt/lists/*

Sym-link in Opencv for virtualenv:
ls -l /usr/local/lib/python2.7/site-packages/
Look for cv2.so

cd ~/.virtualenvs/(virtualenv)/lib/python2.7/site-packages/
ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so

check:
workon (virtualenv)
python
import cv2
cv2.__version__

https://medium.com/@vivek.yadav/deep-learning-setup-for-ubuntu-16-04-tensorflow-1-2-keras-opencv3-python3-cuda8-and-cudnn5-1-324438dd46f0
http://www.pyimagesearch.com/2016/07/11/compiling-opencv-with-cuda-support/
