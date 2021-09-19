# keras-onnx-trt

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

### Demo

```bash
# Convert yolov4 weights to h5 format
python convert.py 

# Convert h5 model to onnx format
python onnx.py

# Below section is about implementing the yolov4 in TensorRT

# Run demo tensorflow
python detect.py

```

### Convert to TensorRT
```bash# yolov3
python save_model.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf --input_size 416 --model yolov3
python convert_trt.py --weights ./checkpoints/yolov3.tf --quantize_mode float16 --output ./checkpoints/yolov3-trt-fp16-416

# yolov3-tiny
python save_model.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --input_size 416 --tiny
python convert_trt.py --weights ./checkpoints/yolov3-tiny.tf --quantize_mode float16 --output ./checkpoints/yolov3-tiny-trt-fp16-416

# yolov4
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4.tf --input_size 416 --model yolov4
python convert_trt.py --weights ./checkpoints/yolov4.tf --quantize_mode float16 --output ./checkpoints/yolov4-trt-fp16-416
```

### Implementing in TensorRT environment
```bash
# Configure TensorRT environment
# To check CUDA version

nvidia-smi

# Installing CUDA and compaitable TensorRT packages

sudo apt-get --purge remove cuda nvidia* libnvidia-*
sudo dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
sudo apt-get remove cuda-*
sudo apt autoremove

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install -y ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get -y install nvidia-driver-418

# Install development and runtime libraries (~4GB)
sudo apt-get install -y cuda-10-0 libcudnn7=7.6.2.24-1+cuda10.0 libcudnn7-dev=7.6.2.24-1+cuda10.0 --allow-change-held-packages
sudo dpkg -i "/content/drive/My Drive/nv-tensorrt-repo-ubuntu1804-cuda10.0-trt7.0.0.11-ga-20191216_1-1_amd64.deb"
sudo apt-key add /var/nv-tensorrt-repo-cuda10.0-trt7.0.0.11-ga-20191216/7fa2af80.pub
sudo apt-get update

sudo apt-get install libnvinfer7=7.0.0-1+cuda10.0 libnvonnxparsers7=7.0.0-1+cuda10.0 libnvparsers7=7.0.0-1+cuda10.0 libnvinfer-plugin7=7.0.0-1+cuda10.0 libnvinfer-dev=7.0.0-1+cuda10.0 libnvonnxparsers-dev=7.0.0-1+cuda10.0 libnvparsers-dev=7.0.0-1+cuda10.0 libnvinfer-plugin-dev=7.0.0-1+cuda10.0 python-libnvinfer=7.0.0-1+cuda10.0 python3-libnvinfer=7.0.0-1+cuda10.0

sudo apt-mark hold libnvinfer7 libnvonnxparsers7 libnvparsers7 libnvinfer-plugin7 libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev python-libnvinfer python3-libnvinfer

# Download TensorRT-5.1.5 from nvidia's official site, untar and install

tar -xvzf TensorRT-5.1.5.0.Ubuntu-18.04.2.x86_64-gnu.cuda-10.0.cudnn7.5.tar.gz
cd TensorRT-5.1.5.0/python && ls && sudo pip3 install tensorrt-5.1.5.0-cp37-none-linux_x86_64.whl
cd TensorRT-5.1.5.0/uff && sudo pip3 install uff-0.6.3-py2.py3-none-any.whl
cd TensorRT-5.1.5.0/graphsurgeon && sudo pip3 install graphsurgeon-0.4.1-py2.py3-none-any.whl

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-5.1.5.0/lib
export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
sudo cp TensorRT-5.1.5.0/targets/x86_64-linux-gnu/lib/lib* /usr/lib/

# Check the tensorrt version
import tensorrt
print(tensorrt.__version__)

sudo apt-get install protobuf-compiler libprotobuf-dev
git clone --recursive https://github.com/onnx/onnx-tensorrt.git
cd onnx-tensorrt && git checkout v5.0 && mkdir build && cd build && cmake .. -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include -DTENSORRT_ROOT=/content/TensorRT-5.1.5.0 -DGPU_ARCHS="61"

sudo apt-get install swig
pip install pycuda

cd onnx-tensorrt && cd build && make -j8 && sudo make install && cd .. && sudo python setup.py install

```

### References
* https://github.com/prratadiya/tensorrt-installation-colab/blob/master/TensorRT_7_0_0_installation.ipynb
* https://forums.developer.nvidia.com/t/how-to-check-my-tensorrt-version/56374/11
* https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-700/tensorrt-install-guide/index.html#downloading
* https://forums.developer.nvidia.com/t/importerror-libnvinfer-so-4-cannot-open-shared-object-file-no-such-file-or-directory/62243/8
* https://developer.nvidia.com/nvidia-tensorrt-5x-download
* https://medium.com/analytics-vidhya/installation-guide-of-tensorrt-for-yolov3-58a89eb984f
* https://github.com/onnx/onnx-tensorrt/issues/126#issuecomment-537892082
