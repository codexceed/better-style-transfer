# Better Neural Style Transfer


Command to run style transfer with default parameters using the figures.jpg content image and the candy.jpg style image:
```
python neural_style_transfer.py --content_img_name figures.jpg \
              --style_img_name candy.jpg
```

Command to run style transfer with Gaussian random noise initialization and custom content/style weights parameters using the figures.jpg content image and the candy.jpg style image:
```
python neural_style_transfer.py --content_img_name figures.jpg \
              --style_img_name candy.jpg \
              --init_method gaussian \
              --content_weight 1e5 \
              --style_weight 1e6 \
```

Command to run style transfer with content/style features coming from custom layers from the InceptionV3 model using the figures.jpg content image and the candy.jpg style image:
```
python neural_style_transfer.py --content_img_name figures.jpg \
              --style_img_name candy.jpg \
              --model inceptionV3 \
              --content_layer 4 \
              --style_layers 6
```
