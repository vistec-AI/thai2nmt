git clone https://github.com/google/sentencepiece.git ./sentencepiece

cd ./sentencepiece

mkdir -p build

cd build

cmake ..

make -j $(nproc)

make install

ldconfig -v