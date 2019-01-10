# vitaFlow Environment Setup

# OS

- Ubuntu 16+

## Python Environment

Python interpreter : Python 3+

Dependency and environment manager : Anaconda

```bash 
# One time setup
conda create -n vitaflow-gpu python3 pip 
source activate vitaflow-gpu
python --version
```

```bash
cd /path/to/repo/vitaFlow
source activate vitaflow-gpu
pip install -r requirements.txt
```

## System Dependencies

- Tesseract
    - [4.0](https://launchpad.net/~alex-p/+archive/ubuntu/tesseract-ocr)
    ```
    sudo add-apt-repository ppa:alex-p/tesseract-ocr
    sudo apt-get update
    sudo apt-get install tesseract-ocr
    ```
