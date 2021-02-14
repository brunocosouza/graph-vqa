## Running the code

### Pre-trained models

```bash
cd LXMERT
ln -s path/to/pvqa/ data/pvqa
cp -r saved/lxmert data/
 ````
 
 The pre-trained model (870 MB) is available at http://nlp.cs.unc.edu/data/model_LXRT.pth, and can be downloaded with:
 ```bash
mkdir snap/pretrained
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
## Test
```bash
cd LXMERT
mkdir -p snap/output_test
python PVQA.py \
      --test test  --train val --valid " " \
      --load snap/output/BEST \
      --llayers 9 --xlayers 5 --rlayers 5 \
      --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
      --tqdm --output snap/output_test
