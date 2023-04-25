Developed on Ubuntu 22.04 with Python 3.10.6 for my CS 4100 Final Project

# Installation Instructions
```
git clone https://github.com/garylvov/CS4100_AI_final.git
pip install numpy
pip install -U scikit-learn
# For GPU training change cu118 to what cuda version you have - you can check with 'nvidia-smi'
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu118/torch_stable.html
pip install torch-summary
pip install matplotlib
```

# Training Neural Network Approach
```
python3 train.py
```

# K-Means clustering analysis
```
python3 clustering.py
```
