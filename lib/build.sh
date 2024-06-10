cp CMakeLists-fast.txt CMakeLists.txt

mkdir build
cd build

cmake ../
make
cp *.so ../

cd -
