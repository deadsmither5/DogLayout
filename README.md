# DogLayout
![Image text](https://github.com/deadsmither5/DogLayout/blob/main/inference.png)
Visualization of DogLayout's inference process. During inference, we first obtain the noisy layout from standard gaussian. Then the generator takes it as input to output the predicted clean layout. Subsequently, we derive the less noisy layout by adding noise to the predicted clean layout.
## code for DogLayout