## Installation

1. Install poppler (for pdfs)
   [Windows]
   https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.08.0-0

[Mac]
brew install poppler

2. Install dependencies
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python app.py

### If CUDA not detected...

```
pip uninstall -y torch torchvision torchaudio
pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126

```

### Server the RAG Server UI

```
docker run -d -p 80:80 -v ./public_html:/usr/share/nginx/html nginx:alpine
```
