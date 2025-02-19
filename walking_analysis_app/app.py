import os
import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from openai import OpenAI
import json

# ✅ 環境変数を読み込む
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ✅ Supabase クライアントを作成
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ OpenAI クライアントを作成
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Mediapipe のセットアップ
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ✅ Webアプリのタイトル
st.title("🚶‍♂️ 歩行分析アプリ")
st.write("動画をアップロードすると歩行を解析します！")

# ✅ 動画アップロード
uploaded_file = st.file_uploader("歩行動画をアップロードしてください", type=["mp4", "mov"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)

    # ✅ 保存用の動画ファイルを作成
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if st.button("歩行解析を開始"):
        st.write("解析中...")

        joint_data = []

        # ✅ 取得する関節
        JOINTS = {
            "LEFT_HIP": mp_pose.PoseLandmark.LEFT_HIP,
            "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP,
            "LEFT_KNEE": mp_pose.PoseLandmark.LEFT_KNEE,
            "RIGHT_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE,
            "LEFT_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE,
            "RIGHT_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE
        }

        # ✅ Pose モデルを初期化
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:
                    frame_data = {"Time (s)": cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FPS)}
                    for joint_name, joint_id in JOINTS.items():
                        landmark = results.pose_landmarks.landmark[joint_id]
                        frame_data[f"{joint_name}_Y"] = landmark.y

                    joint_data.append(frame_data)

                    # ✅ 関節マーカーを描画
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # ✅ フレームを動画に保存
                out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()

        df = pd.DataFrame(joint_data)
        st.write("✅ 解析完了！")

        # ✅ 歩行バランスのグラフを表示
        fig = px.line(df, x="Time (s)", y=["LEFT_KNEE_Y", "RIGHT_KNEE_Y", "LEFT_ANKLE_Y", "RIGHT_ANKLE_Y"],
                      title="歩行バランスの変化", labels={"value": "関節の高さ", "variable": "関節"})
        st.plotly_chart(fig)

        # ✅ 解析動画を表示
        st.subheader("🎥 解析結果の動画")
        st.video(output_video_path)

        # ✅ 解析動画をダウンロード
        with open(output_video_path, "rb") as file:
            st.download_button("📥 解析動画をダウンロード", file, file_name="walking_analysis.mp4", mime="video/mp4")

        # ✅ 歩行スコアを計算
        def calculate_gait_scores(df):
            scores = {}
            scores["Stability Score"] = max(0, 100 - (df["LEFT_KNEE_Y"].std() + df["RIGHT_KNEE_Y"].std()) * 50)
            step_intervals = np.diff(df["Time (s)"])
            scores["Gait Rhythm Score"] = max(0, 100 - np.std(step_intervals) * 500)
            scores["Symmetry Score"] = max(0, 100 - np.mean(np.abs(df["LEFT_KNEE_Y"] - df["RIGHT_KNEE_Y"])) * 500)
            return scores

        scores = calculate_gait_scores(df)
        st.metric(label="歩行安定度スコア", value=f"{scores['Stability Score']:.1f} / 100")

        # ✅ AI に解析データを送信し、解説を取得
        def generate_ai_analysis(scores_json):
            prompt = f"""
            あなたは歩行解析の専門家です。
            以下の解析結果をわかりやすく解説してください：
            {json.dumps(scores_json, indent=2, ensure_ascii=False)}
            どのスコアが良く、どのスコアが改善の余地があるか説明してください。
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは歩行解析の専門家です。"},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content

        ai_analysis = generate_ai_analysis(scores)

        st.subheader("📖 AI による解析解説")
        st.write(ai_analysis)

        # ✅ Supabase にデータを保存
        supabase.table("walking_analysis").insert({
            "scores": json.dumps(scores),
            "ai_analysis": ai_analysis
        }).execute()

        st.success("データをクラウドに保存しました！")

