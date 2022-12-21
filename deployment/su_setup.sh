add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install python3.9 -y
apt-get install python3-pip  -y
apt-get install python-is-python3 -y
apt-get install -y mesa-vulkan-drivers -y
apt-get install ffmpeg libsm6 libxext6  -y
pip install -r requirements.txt
python vulkanese/math/loiacono/loiacono_gpu.py
