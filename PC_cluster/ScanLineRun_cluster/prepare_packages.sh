mkdir third_party
cd third_party
git clone https://github.com/pybind/pybind11.git
git clone https://gitlab.com/libeigen/eigen.git

apt install libopencv-dev
apt-get install wget

apt-get install doxygen
apt-get install mpi-default-dev openmpi-bin openmpi-common
apt-get install libflann1.8 libflann-dev
apt-get install libeigen3-dev
apt-get install libboost-all-dev
apt-get install libvtk6-dev libvtk6.2 libvtk6.2-qt
apt-get install 'libqhull*'
apt-get install libusb-dev
apt-get install libgtest-dev
apt-get install git-core freeglut3-dev pkg-config
apt-get install build-essential libxmu-dev libxi-dev
apt-get install libusb-1.0-0-dev graphviz mono-complete
apt-get install qt-sdk openjdk-9-jdk openjdk-9-jre
apt-get install phonon-backend-gstreamer
apt-get install phonon-backend-vlc
apt-get install libflann-dev
apt-get install libopenni-dev libopenni2-dev

# Compile and install PCL
wget https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.11.1.tar.gz
tar xvf pcl-1.11.1.tar.gz
cd pcl-pcl-1.11.1 && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2
make -j2 install
cd ..