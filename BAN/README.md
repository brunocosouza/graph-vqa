# Implementation of Method 2
Modified from [Bilinear Attention Networks](http://arxiv.org/abs/1805.07932).  


```bash
ln -s 'C:\Users\bruno\Projetos\graph-based-vqa\PVQA Baseline Dataset\LXMERT\data\pvqa' 'C:\Users\bruno\Projetos\graph-based-vqa\PVQA Baseline Dataset\BAN\data'

python .\finetune_main.py --task pvqa --epoch 1 --start_epoch 0 --lr 0.01 --cos --train train --val val --tfidf --output saved_models/pvqa/pre --batch_size 128
```
