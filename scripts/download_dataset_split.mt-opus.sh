VERSION=${1:-"1.0"}

echo "English-Thai MT Dataset train/val/test split (mt-opus), available at https://github.com/vistec-AI/thai2nmt/releases"
echo "Release version: $VERSION (default: 1.0)"

echo "Downloading the csv file from the URL below and save files to '../dataset/split/mt-opus/'"

echo "https://github.com/vistec-AI/thai2nmt/releases/download/mt-opus_v${VERSION}/en-th.merged_stratified.train.csv"
echo "https://github.com/vistec-AI/thai2nmt/releases/download/mt-opus_v${VERSION}/en-th.merged_stratified.val.csv"
echo "https://github.com/vistec-AI/thai2nmt/releases/download/mt-opus_v${VERSION}/en-th.merged_stratified.test.csv"

mkdir -p ./dataset/split/mt-opus

curl -Lk https://github.com/vistec-AI/thai2nmt/releases/download/mt-opus_v${VERSION}/en-th.merged_stratified.train.csv \
     -o ./dataset/split/mt-opus/en-th.merged_stratified.train.csv

curl -Lk https://github.com/vistec-AI/thai2nmt/releases/download/mt-opus_v${VERSION}/en-th.merged_stratified.val.csv \
     -o ./dataset/split/mt-opus/en-th.merged_stratified.val.csv

curl -Lk https://github.com/vistec-AI/thai2nmt/releases/download/mt-opus_v${VERSION}/en-th.merged_stratified.test.csv \
     -o ./dataset/split/mt-opus/en-th.merged_stratified.test.csv

echo "Done."

