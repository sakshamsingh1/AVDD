
echo "Downloading VGG subset IPC1 data"
wget -P data/syn_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/syn_data/VGG_subset_ipc1.zip
echo "Unzipping VGG subset IPC1 data"
unzip data/syn_data/VGG_subset_ipc1.zip -d data/syn_data/
echo "Removing zip file"
rm data/syn_data/VGG_subset_ipc1.zip

echo "Downloading VGG subset IPC10 data"
wget -P data/syn_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/syn_data/VGG_subset_ipc10.zip
echo "Unzipping VGG subset IPC10 data"
unzip data/syn_data/VGG_subset_ipc10.zip -d data/syn_data/
echo "Removing zip file"
rm data/syn_data/VGG_subset_ipc10.zip

echo "Downloading VGG subset IPC20 data"
wget -P data/syn_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/syn_data/VGG_subset_ipc20.zip
echo "Unzipping VGG subset IPC20 data"
unzip data/syn_data/VGG_subset_ipc20.zip -d data/syn_data/
echo "Removing zip file"
rm data/syn_data/VGG_subset_ipc20.zip

echo "Downloading VGG subset train/test data"
wget -P data/test_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/test_data/vgg_subset_test.pt