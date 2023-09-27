# GitHub Link
https://github.com/BharathChalla/Efficient-Prompt.git

# Conda Environment Setup
conda create -n effprompt python=3.9
conda activate effprompt

## As per README.md Dependencies 
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorboardX einops tqdm 

## Other Dependencies
pip install opencv-python pandas numpy pytorchvideo timm ftfy
pip install transformers sentence-transformers
