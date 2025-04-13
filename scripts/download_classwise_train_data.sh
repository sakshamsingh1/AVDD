# VGGSubset
echo "Downloading classwise VGGSubset train data"
wget -P data/classwise_train_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/classwise_train_data/class_wise_vgg_subset.zip
echo "Unzipping classwise VGGSubset train data"
unzip data/classwise_train_data/class_wise_vgg_subset.zip -d data/classwise_train_data/
echo "Removing zip file"
rm data/classwise_train_data/class_wise_vgg_subset.zip

# Uncomment the following lines to download other datasets
# # VGG
# echo "Downloading classwise VGG train data"
# wget -P data/classwise_train_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/classwise_train_data/class_wise_vgg.zip
# echo "Unzipping classwise VGG train data"
# unzip data/classwise_train_data/class_wise_vgg.zip -d data/classwise_train_data/
# echo "Removing zip file"
# rm data/classwise_train_data/class_wise_vgg.zip


# # AVE
# echo "Downloading classwise AVE train data"
# wget -P data/classwise_train_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/classwise_train_data/class_wise_ave.zip
# echo "Unzipping classwise AVE train data"
# unzip data/classwise_train_data/class_wise_ave.zip -d data/classwise_train_data/
# echo "Removing zip file"
# rm data/classwise_train_data/class_wise_ave.zip

# # Music 21
# echo "Downloading classwise Music train data"
# wget -P data/classwise_train_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/classwise_train_data/class_wise_music_21.zip
# echo "Unzipping classwise Music train data"
# unzip data/classwise_train_data/class_wise_music_21.zip -d data/classwise_train_data/
# echo "Removing zip file"
# rm data/classwise_train_data/class_wise_music_21.zip


# # VGGSubset_10s
# echo "Downloading classwise VGGSubset 10s train data"
# wget -P data/classwise_train_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/classwise_train_data/class_wise_vgg_subset_10s.zip
# echo "Unzipping classwise VGGSubset 10s train data"
# unzip data/classwise_train_data/class_wise_vgg_subset_10s.zip -d data/classwise_train_data/
# echo "Removing zip file"
# rm data/classwise_train_data/class_wise_vgg_subset_10s.zip

# # AVE_10s
# echo "Downloading classwise AVE 10s train data"
# wget -P data/classwise_train_data/ https://huggingface.co/datasets/sakshamsingh1/AVDD_data/resolve/main/classwise_train_data/class_wise_ave_10s.zip
# echo "Unzipping classwise AVE 10s train data"
# unzip data/classwise_train_data/class_wise_ave_10s.zip -d data/classwise_train_data/
# echo "Removing zip file"
# rm data/classwise_train_data/class_wise_ave_10s.zip