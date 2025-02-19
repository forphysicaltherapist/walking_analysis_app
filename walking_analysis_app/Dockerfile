# ベースイメージ
FROM python:3.10-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要なシステムパッケージのインストール
# libgl1-mesa-glx および libglib2.0-0 を追加
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 必要な Python パッケージのインストール
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# アプリのソースコードをコピー
COPY . /app

# アプリの起動
CMD ["streamlit", "run", "walking_analysis_app/app.py"]
