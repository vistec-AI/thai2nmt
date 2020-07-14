GPU=${1:-"n"}

echo "Clone GitHub repo from https://github.com/pytorch/fairseq"

git clone https://github.com/pytorch/fairseq.git ./fairseq

cd ./fairseq

echo "Reset to the commit ID #6f6461b on 'master' branch"

git reset --hard 6f6461b 

echo "Install fairseq from source"

pip install --editable .

cd ..

echo "System argument, GPU, is ${GPU} (default: 'n')"

if [ $GPU = 'y' ]
then

    echo "Install 'apex' library"
    
    git clone https://github.com/NVIDIA/apex ./apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
        --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
        --global-option="--fast_multihead_attn" ./

fi

echo "Done."