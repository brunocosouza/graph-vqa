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
python pvqa.py \
      --train train --valid val  \
      --llayers 9 --xlayers 5 --rlayers 5 \
      --loadLXMERT snap/pretrained/ \
      --batchSize 32 --optim bert --lr 5e-5 --epochs 5 \
      --tqdm --output snap/output
```