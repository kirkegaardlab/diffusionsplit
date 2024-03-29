# Diffusion Splitting for Instance Segmentation
<p align="center">
  <img src="anim.gif" height="512" />
</p>

Paper: [https://www.biorxiv.org/content/10.1101/2023.07.07.548066v1](https://www.biorxiv.org/content/10.1101/2023.07.07.548066v1)

### Installation
```
pip install -r requirements.txt
pip install https://github.com/zsef123/Connected_components_PyTorch/archive/refs/heads/main.zip
```

### Dataset
Overlapping dataset can be downloaded at
[github.com/juliusbierk/cellpose-overlap](https://github.com/juliusbierk/cellpose-overlap).

### Training
```python
python split.py --data data/overlapdataset --model overlapdataset --train-all
```

download pretrained:
```bash
mkdir models
mkdir models/overlapdataset
cd models/overlapdataset
wget https://sid.erda.dk/share_redirect/AvuJczAyUI -O overlap_trained.zip
unzip overlap_trained.zip
```

### Evaluation
```python
python split.py --data data/overlapdataset --model overlapdataset --eval
```
