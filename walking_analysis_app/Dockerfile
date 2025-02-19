# ベースイメージ
FROM python:3.10-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要なパッケージのインストール
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    pip install --no-cache-dir numpy==1.23.5 opencv-python-headless==4.5.5.64 Pillow==9.5.0 streamlit

# アプリのソースコードをコピー
COPY . /app

# アプリの起動
CMD ["streamlit", "run", "app.py"]
