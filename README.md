To reproduce results:

First install sslime and download STL-10, then install visdom:
```
pip install visdom
```

each experiment has its own yaml file. to run yaml file,  

To train the models for our best model (VGG19)

```
PYTHONPATH=fair-sslime python fair-sslime/tools/train.py --config_file fair-sslime/configs/final_eval/final_eval_3.yaml
PYTHONPATH=fair-sslime python fair-sslime/tools/train.py --config_file fair-sslime/configs/final_pretext/final_pretext_13.yaml
```

For other experiments, yaml files will be in the following folders:

Downstream evaluation:
fair-sslime/configs/final_eval/

Pretext tasks:
fair-sslime/configs/final_pretext/

Preprocess screening:
fair-sslime/configs/preprocess

link to model checkpoints: 
  VGG: https://drive.google.com/open?id=16XQuysAAn0n56JSL1j-m7_91iMdtABCz
  
to see our implementation of the TRUNK for the jigsaw model:
fair-sslime/sslime/models/trunks/alexnet_jigsaw.py

to see our implementation of the TRUNK for the revnet model:
fair-sslime/sslime/models/trunks/revnet50.py
