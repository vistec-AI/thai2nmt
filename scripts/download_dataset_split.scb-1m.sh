VERSION=${1:-"1.0"}

echo "English-Thai MT Dataset train/val/test split (scb-mt-en-th-2020), available at https://github.com/vistec-AI/thai2nmt/releases"
echo "Release version: $VERSION (default: 1.0)"

echo "Downloading the csv file from the URL below and save files to '../dataset/split/scb-mt-en-th-2020/'"

echo "https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020_v${VERSION}/en-th.merged_stratified.train.csv"
echo "https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020_v${VERSION}/en-th.merged_stratified.val.csv"
echo "https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020_v${VERSION}/en-th.merged_stratified.test.csv"

mkdir -p ./dataset/split/scb-mt-en-th-2020

curl -Lk https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020_v${VERSION}/en-th.merged_stratified.train.csv\
     -o ./dataset/split/scb-mt-en-th-2020/en-th.merged_stratified.train.csv

curl -Lk https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020_v${VERSION}/en-th.merged_stratified.val.csv\
     -o ./dataset/split/scb-mt-en-th-2020/en-th.merged_stratified.val.csv

curl -Lk https://github.com/vistec-AI/thai2nmt/releases/download/scb-mt-en-th-2020_v${VERSION}/en-th.merged_stratified.test.csv\
     -o ./dataset/split/scb-mt-en-th-2020/en-th.merged_stratified.test.csv

echo "Done."

