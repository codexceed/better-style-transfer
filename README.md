# Better Neural Style Transfer


```
python neural_style_transfer.py --content_img_name figures.jpg \
              --style_img_name candy.jpg
```

```
python neural_style_transfer.py --content_img_name figures.jpg \
              --style_img_name candy.jpg \
              --init_method gaussian \
              --content_weight 1e5 \
              --style_weight 1e6 \
              --tv_weight 1e0 \
```


```
python neural_style_transfer.py --content_img_name figures.jpg \
              --style_img_name candy.jpg \
            --init_method content \
            --model inceptionV3 \
            --content_layer 4 \
            --style_layers 6
```
