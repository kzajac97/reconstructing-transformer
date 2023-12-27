# reconstructing-transformer

Set of notebooks used to demonstrate how transformers work using code and real data.

All runs are added to W&B project: [https://wandb.ai/kzajac/vision-transformer](https://wandb.ai/kzajac/vision-transformer).

### Notebooks

* `encoder-decoder-for-machine-translation.ipynb` - using vanilla architecture of encoder-decoder transformer for German to English machine translation
* `reconstructing-transformer.ipynb` - notebook with exploration of transformer code connected to equations from the paper
* `vision-transformer-for-mnist.ipynb` - notebook with minimal implementation of Vision Transformer (ViT) for image classification using MNIST

### Requirements

The repository contains docker multiple files, which can be used to build docker images to run different notebooks or 
parts of code. All images are stored in `docker` directory, each with its own requirements. Current images:
* `generic` - running notebooks and visualizing results (CPU only), for example `reconstructing-transformer.ipynb`
* `vit` - running examples of VIT training
* `llama` - running examples of LLAMA-2 fine-tuning

# Contents

## Vision Transformer

Vision transformer model is implemented in the source code package. Training and evaluation is done in the notebook `vision-transformer-for-mnist.ipynb`,
which can be run for test purpose with different model parameters. Non-exhaustive experiments show following results.

Baselines are added to the `models.py` file and executed using the same notebook as ViT models. For the MNIST problem,
CNN model is performing better than ViT models, but the problem is simple and can be considered solved with all models 
having over 97% accuracy. The implementation of VIT on such dataset serves mostly as didactic example.

| Model                | Number of Parameters | Accuracy | F1Score | Training Time |
|----------------------|----------------------|----------|---------|---------------|
| CNN Baseline         | 1199882              | 0.9894   | 0.9894  | 3m 05 s       |
| Deep ViT             | 502346               | 0.9838   | 0.9836  | 6m 56 s       |
| Large ViT            | 1327050              | 0.9825   | 0.9824  | 5m 53 s       |
| Feedforward Baseline | 104938               | 0.9728   | 0.9725  | 2m 52 s       |
| Small ViT            | 150490               | 0.9703   | 0.9701  | 4m 56 s       |
