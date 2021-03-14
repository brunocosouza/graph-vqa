# graph-vqa


## Dataset
### Prepare PathVQA dataset:  
Download Dataset from : Google Drive Link: https://drive.google.com/file/d/1utnisF_HJ8Yk9Qe9dBe9mxuruuGe7DgW/view?usp=sharing

```bash
cd LXMERT
mkdir data
ln -s path/to/pvqa/ data/pvqa
cp -r saved/lxmert data/
```
cd
## Running the code

### Pre-trained models

The pre-trained model (870 MB) is available at http://nlp.cs.unc.edu/data/model_LXRT.pth, and can be downloaded with:

```bash
cd LXMERT
cd snap/pretrained 
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```

Run :
```bash
cd LXMERT
python PVQA.py \
      --train train --valid val  \
      --llayers 9 --xlayers 5 --rlayers 5 \
      --loadLXMERT snap/pretrained/model \
      --batchSize 32 --optim bert --lr 5e-5 --epochs 20 \
      --tqdm --output snap/output
```

```bash
cd LXMERT
mkdir -p snap/output_test
python PVQA.py \
      --test test  --train val --valid " " \
      --load snap/output/BEST \
      --llayers 9 --xlayers 5 --rlayers 5 \
      --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
      --tqdm --output snap/output_test


  # Pre-training
python src/pretrain/lxmert_pretrain.py \
      --taskQA_woi --taskVA2 --taskMatched --taskQA \
      --visualLosses obj,attr,feat \
      --wordMaskRate 0.15 --objMaskRate 0.15 \
      --train  pvqa_train --valid pvqa_val \
      --llayers 9 --xlayers 5 --rlayers 5 \
      --batchSize 16 --optim bert --lr 1e-4 --epochs 2 \
      --tqdm --output snap/output_pretraining