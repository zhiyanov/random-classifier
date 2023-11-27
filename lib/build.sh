mkdir build
cd build

cmake ../
make
cp *.so ../

cd -
rm -rf build
