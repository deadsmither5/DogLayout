# DogLayout: Denoising Diffusion GAN for Discrete and Continuous Layout Generation
![Image text](https://github.com/deadsmither5/DogLayout/blob/main/inference.png)
Visualization of DogLayout's inference process. During inference, we first obtain the noisy layout from standard gaussian. Then the generator takes it as input to output the predicted clean layout. Subsequently, we derive the less noisy layout by adding noise to the predicted clean layout.
## Abstract 
Layout Generation aims to synthesize plausible arrangements from given elements. DogLayout (Denoising Diffusion GAN Layout model), which integrates a diffusion process into GANs to enable the generation of discrete label data and significantly reduce diffusion’s sampling time.
## Dataset
The datasets are available at: 
```
wget https://huggingface.co/datasets/puar-playground/LACE/resolve/main/datasets.tar.gz 
tar -xvzf datasets.tar.gz
```

PubLayNet: Download the labels.tar.gz and decompress to ./dataset/publaynet-max25/raw folder.  
Rico: Download the rico_dataset_v0.1_semantic_annotations.zip and decompress to ./dataset/rico25-max25/raw folder.  

## Training 
For PubLayNet:
```
python train_all.py --dataset publaynet --device cuda:0 --G_d_model  512 --G_nhead 8 --G_num_layers  4 --D_d_model  512 --D_nhead  8 --D_num_layers  4 --num_timesteps  8 --latent_dim  4 --num_label  6
```
For Rico:
```
python train_all.py --dataset rico25 --device cuda:0 --G_d_model  512 --G_nhead  8 --G_num_layers  4 --D_d_model  512 --D_nhead  8 --D_num_layers  4 --num_timesteps  8 --latent_dim  4 --num_label  26 
```
## Testing
For PubLayNet:
```
python test.py --dataset publaynet --experiment all --device cuda:0 --G_d_model  512 --G_nhead  8 --G_num_layers  4 --D_d_model  512 --D_nhead  8 --D_num_layers  4 --num_timesteps  8 --latent_dim  4 --num_label  6 
```
For Rico:
```
python test.py --dataset rico25 --experiment all --device cuda:0 --G_d_model  512 --G_nhead  8 --G_num_layers  4 --D_d_model  512 --D_nhead  8 --D_num_layers  4 --num_timesteps  8 --latent_dim  4 --num_label  26 
```