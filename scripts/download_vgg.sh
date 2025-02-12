echo "Downloading VGG IPC1 data"
wget -P data/syn_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/syn_data/vgg_ipc1.zip
echo "Unzipping VGG IPC1 data"
unzip data/syn_data/vgg_ipc1.zip -d data/syn_data/
echo "Removing zip file"
rm data/syn_data/vgg_ipc1.zip

echo "Downloading VGG IPC20 data"
wget -P data/syn_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/syn_data/vgg_ipc20.zip
echo "Unzipping VGG IPC20 data"
unzip data/syn_data/vgg_ipc20.zip -d data/syn_data/
echo "Removing zip file"
rm data/syn_data/vgg_ipc20.zip

echo "Downloading VGG test data"
wget -P data/test_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/test_data/vgg_test.pt
