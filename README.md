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
Note: Please uncomment different commands in the file to test different IPC settings

### Visualization
You can visualize the distilled data using the `visualize_data.ipynb` notebook. 

## TODO

- [x] Inference code
- [ ] Training code
- [ ] Parallelized training code
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