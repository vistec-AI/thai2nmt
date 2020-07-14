VERSION=${1:-"1.0"}

echo "English-Thai MT Dataset (scb-mt-en-th-2020), available at https://github.com/vistec-AI/thai2nmt/releases"
echo "Release version: $VERSION (default: 1.0)"

echo "Downloading the zipped file and save to '../dataset/raw/'"
echo "https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020_v${VERSION}/scb-mt-en-th-2020.tar.gz"

curl -Lk https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020_v${VERSION}/scb-mt-en-th-2020.tar.gz \
     -o ./dataset/raw/scb-mt-en-th-2020.tar.gz

echo "Extracting .tar.gz file "

tar -xvzf ./dataset/raw/scb-mt-en-th-2020.tar.gz  -C ./dataset/raw/

rm ./dataset/raw/scb-mt-en-th-2020.tar.gz

echo "Done."

