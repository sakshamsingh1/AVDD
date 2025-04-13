# AVDD

This repository includes code for : [Audio Visual Dataset Distillation (TMLR 2024)](https://openreview.net/pdf?id=IJlbuSrXmk).

## Create env
```bash
conda create -n avdd python=3.9 -y
conda activate avdd
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/sakshamsingh1/AVDD
cd AVDD
pip install -r requirements.txt
```

## Inference code
Download and unzip VGG-10k distilled and real data.
```bash
bash scripts/download_vgg_subset.sh
```

**Evaluate VGG-10k data.** The code will evaluate 3 distilled data for 5 times.  
```bash
bash scripts/vgg10k_evaluate.sh
```
Note: Please uncomment different commands in the file to test different IPC settings \
Similar scripts are present for other datasets.

**Visualization** \
You can visualize the distilled data using the `visualize_data.ipynb` notebook. 

## Training

### Prepare data
Download preprocessed training data from huggingface
```bash
# Open the script and uncomment the datasets you wish to download
bash scripts/download_train_data.sh
```
<details style="margin-top: -8px;">
  <summary>Preparing preprocessed training/testing data (For eg. AVE)</summary>
  <ul>
    <li>Download <a href="https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view" target="_blank">AVE</a> dataset </li>
    <li>Extract audio/frames <code>preprocess/extract_audio_and_frames.py</code> </li>
    <li>Create training/testing dataset (.pt) file <code>preprocess/AVE_input_data.py</code> ( for VGG-subset see:  <code>preprocess/VGG_subset_input_data.py</code>) </li>
    <li>These scripts can be modified to support other datasets. </li>
    <li>We also provide metadata in <code>preprocess/meta_data</code> . </li>
  </ul>
</details>

### Herding
We provide the precomputed herding indices in `data/herding_data`

<details style="margin-top: -10px;">  
  <summary> More </summary>
  <ul>
  <li>The synthetic data is initialised with herding selected method.</li>
  <li> To compute herding data, we follow the pseudocode mentioned <a href="https://github.com/VICO-UoE/DatasetCondensation/issues/15#issuecomment-1242561403" target="_blank"> here </a>.</li>
  </ul>
</details>

### Launch training
Download preprocessed training data from huggingface
```bash
# Please open the script and refer to some examples here.
bash scripts/train_avdd.sh
```

## Classwise training code
<details style="margin-top: -5px;">  
  <summary> More </summary>
  <ul>
    <li>Synthetic data can be learned independently for each class.</li>
    <li>Training all classes together with high class counts and images-per-class (IPC) may lead to out-of-memory (OOM) issues.</li>
    <li>This approach allows parallel training for each class.</li>
  </ul>
</details>

### Download classwise training data
```bash
# Open the script and uncomment the datasets you wish to download 
bash scripts/download_classwise_train_data.sh
```
Alternatively, you can generate class-wise data from the full training set using: ` python preprocess/create_classwise_data.py`

### Pre-saving real mean embeddings
```bash
# Basically we can pre-save real mean embeddings with ipc=1
bash scripts/create_classwise_buffer.sh
```
Alternatively, you can download pre-extracted embeddings from [huggingface](https://huggingface.co/datasets/sakshamsingh1/AVDD_data/tree/main/buffers)

### Learn synthetic data classwise
Now you can learn the synthetic data classwise with different IPCs
```bash
bash scripts/train_classwise.sh
```

### Evaluate classwise trained data
```bash
bash scripts/classwise_evaluate.sh
```

## TODO

- [x] Inference code
- [x] Training code
- [x] Parallelized training code
- [ ] Retrieval code

## 🤗 Citation

```
@article{kushwahaaudio,
  title={Audio-Visual Dataset Distillation},
  author={Kushwaha, Saksham Singh and Vasireddy, Siva Sai Nagender and Wang, Kai and Tian, Yapeng},
  journal={Transactions on Machine Learning Research}
}
```

The code is based on [Distribution Matching](https://github.com/VICO-UoE/DatasetCondensation), [AV-robustness](https://github.com/YapengTian/AV-Robustness-CVPR21)