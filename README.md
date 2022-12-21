# Better Neural Style Transfer
## Introduction
This project aims to replicate the work of [Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) 
on using neural nets to perform **image style transfer** while discovering insights into performance via analysis of the transfer process across various training configurations.

## Setup

1. Install requirements (python 3.8):
  ```shell
  pip3 install requirements.txt
  ```

## Usage 
1. Start the microservice
  ```shell
  $ python run.py 
  * Serving Flask app 'nst_app' (lazy loading)
  * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
  * Debug mode: on
  * Restarting with stat
  * Debugger is active!
  * Debugger PIN: 415-312-085
  * Running on http://127.0.0.1:5001/ (Press CTRL+C to quit)
  ```
2. Check if service is up
  ```shell
  curl -X GET http://127.0.0.1:5001/status
  ```
3. Trigger stylization training and obtain your output token.
  ```shell
  $ curl -X POST -d "content=figures.jpg&style=vg_olive.jpg" http://127.0.0.1:5001/stylize
  36969f13-e2e3-47a0-bf6a-537c91cb3e2a
  ```
4. After some time (depending on your hardware), fetch the output image. In this case, the output image is named "stylized.jpg"
  ```shell
  $ curl -X POST -d "img_id=36969f13-e2e3-47a0-bf6a-537c91cb3e2a" -o stylized.jpg http://127.0.0.1:5001/get-style
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                   Dload  Upload   Total   Spent    Left  Speed
  100  210k  100  210k  100    43   9.7M   2047 --:--:-- --:--:-- --:--:--  9.7M
  ```

## Acknowledgements
### Useful Repositories
- [pytorch-neural-style-transfer](https://github.com/gordicaleksa/pytorch-neural-style-transfer)
- [fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style) (PyTorch, feed-forward method)

### References
- Gatyls et al., [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- [Neural Style Transfer Using PyTorch](https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa)
