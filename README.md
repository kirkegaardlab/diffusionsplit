# Diffusion Splitting for Instance Segmentation
<p align="center">
  <img src="anim.webm" height="512" />
</p>

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
