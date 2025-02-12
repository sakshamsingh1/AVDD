echo "Downloading AVE IPC10 data"
wget -P data/syn_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/syn_data/ave_ipc10.zip
echo "Unzipping AVE IPC10 data"
unzip data/syn_data/ave_ipc10.zip -d data/syn_data/
echo "Removing zip file"
rm data/syn_data/ave_ipc10.zip

echo "Downloading AVE IPC20 data"
wget -P data/syn_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/syn_data/ave_ipc20.zip
echo "Unzipping AVE IPC20 data"
unzip data/syn_data/ave_ipc20.zip -d data/syn_data/
echo "Removing zip file"
rm data/syn_data/ave_ipc20.zip

echo "Downloading AVE test data"
wget -P data/test_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/test_data/ave_test.pt
