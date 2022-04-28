# PixelProcessingPipeline



The very first time you set up your virtual environment, follow these steps:
virtualenv ~/pipeline
source ~/pipeline/bin/activate
pip install scipy ruamel.yaml ibllib PyWavelets scikit-image
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


Every time you start a new session, you will have to activate your virtual environement with
source ~/pipeline/bin/activate