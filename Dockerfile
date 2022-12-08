FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
COPY . /app
WORKDIR /app
RUN apt update && apt install --yes libgl1-mesa-dev libgtk2.0-dev
RUN pip3 install opencv_python-4.6.0.66-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip3 install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "run:app"]