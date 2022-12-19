add-apt-repository ppa:deadsnakes/ppa
apt-get update -y
apt-get install python3.9
apt-get install python3-pip 
apt-get install python-is-python3
apt-get install -y mesa-vulkan-drivers
apt-get install ffmpeg libsm6 libxext6  -y
pip install -r -q requirements.txt
python vulkanese/math/loiacono/loiacono_gpu.py
