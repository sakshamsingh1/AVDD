echo "Downloading Music 21 IPC1 data"
wget -P data/syn_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/syn_data/music_ipc1.zip
echo "Unzipping Music 21 IPC1 data"
unzip data/syn_data/music_ipc1.zip -d data/syn_data/
echo "Removing zip file"
rm data/syn_data/music_ipc1.zip

echo "Downloading Music 21 IPC10 data"
wget -P data/syn_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/syn_data/music_ipc10.zip
echo "Unzipping Music 21 IPC10 data"
unzip data/syn_data/music_ipc10.zip -d data/syn_data/
echo "Removing zip file"
rm data/syn_data/music_ipc10.zip

echo "Downloading Music 21 IPC20 data"
wget -P data/syn_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/syn_data/music_ipc20.zip
echo "Unzipping Music 21 IPC20 data"
unzip data/syn_data/music_ipc20.zip -d data/syn_data/
echo "Removing zip file"
rm data/syn_data/music_ipc20.zip

echo "Downloading Music 21 test data"
wget -P data/test_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/test_data/music_21_test.pt
