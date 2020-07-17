VERSION=${1:-"1.0"}

echo "English-Thai MT Dataset (mt-opus), available at https://github.com/vistec-AI/thai2nmt/releases"
echo "Release version: $VERSION (default: 1.0)"

echo "Downloading the zipped file and save to '../dataset/raw/'"
echo "https://github.com/vistec-AI/thai2nmt/releases/download/mt-opus_v${VERSION}/mt-opus.tar.gz"

curl -Lk https://github.com/vistec-AI/thai2nmt/releases/download/mt-opus_v${VERSION}/mt-opus.tar.gz \
     -o ./dataset/raw/mt-opus.tar.gz

echo "Extracting .tar.gz file "

tar -xvzf ./dataset/raw/mt-opus.tar.gz  -C ./dataset/raw/

rm ./dataset/raw/mt-opus.tar.gz

echo "Done."