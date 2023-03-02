sudo apt-get update -y
sudo apt install python3-pip 
sudo apt install python-is-python3
sudo apt-get install -y mesa-vulkan-drivers
sudo apt-get install ffmpeg libsm6 libxext6  -y
pip install -r -q requirements.txt
python vulkanese/math/loiacono/loiacono_gpu.py
