import os
import time
from flask import Flask, render_template, Response, jsonify, request
from waitress import serve
import cv2
from ultralytics import YOLO
import threading
from datetime import datetime
import numpy as np
from collections import deque  # 録画用バッファ用に追加
import humanize
from flask import send_from_directory

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
app = Flask(__name__)
model = YOLO("yolo11n.pt")


class FrameGenerator:
    """カメラから映像を取得し、YOLO推論結果を提供するクラス."""

    def __init__(self, camera: int = 0, frame_limit: int = 15, frame_size: tuple = (640, 480), quality: int = 85):
        """
        初期化する.
        Args:
            camera (int): 利用するカメラ番号。デフォルトは0。
            frame_limit (int): フレームレート上限。デフォルトは15。
            frame_size (tuple): 画像サイズ。デフォルトは(640, 480)。
            quality (int): JPEGエンコード画質。デフォルトは85。
        """
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise Exception("Camera not found.")
        self.cap.set(cv2.CAP_PROP_FPS, frame_limit)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
        self.frame = None
        self.raw_frame = None  # キャプチャした生フレームを保持
        self.lock = threading.Lock()
        self.running = True
        self.frame_limit = frame_limit
        self.frame_size = frame_size
        self.cache_duration = 1 / self.frame_limit  # キャッシュの持続時間
        self.quality = quality  # 画質の設定
        self.detect_count = 0
        # 追加: 録画用バッファと設定、状態管理
        self.record_buffer = deque(maxlen=int(self.frame_limit * 15))  # 過去15秒分のフレームを保持
        self.record_enabled = True  # 録画設定が有効
        self.recording = False  # 現在録画中か否か
        self.record_save_path = "./records"  # 録画保存先
        self.record_save_path = os.path.abspath(self.record_save_path)  # 絶対パスに変換
        os.makedirs(self.record_save_path, exist_ok=True)  # 保存先ディレクトリの作成
        # カメラ読み込みと推論を分離して並列実行
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.predict_thread = threading.Thread(target=self.predict_loop)
        self.record_monitor_thread = threading.Thread(target=self.record_monitor)
        self.capture_thread.start()
        self.predict_thread.start()
        self.record_monitor_thread.start()

    def capture_loop(self):
        """
        カメラから生フレームを取得し、raw_frameに保持する.
        """
        while self.running:
            success, frame = self.cap.read()
            if not success:
                continue
            # リサイズ
            frame = cv2.resize(frame, self.frame_size)
            with self.lock:
                self.raw_frame = frame
                self.record_buffer.append(frame)  # 録画用バッファへ追加
            time.sleep(0)  # 他スレッドへのCPU譲渡

    def predict_loop(self):
        """
        最新のraw_frameに対してYOLO推論を実行し、検出結果を更新する.
        """
        while self.running:
            with self.lock:
                raw = self.raw_frame
            if raw is not None:
                results = model.predict(raw, save=False, conf=0.3, classes=[0], verbose=False)  # 推論実行
                self.detect_count = results[0].boxes.data.shape[0]  # 検出オブジェクト数更新
                with self.lock:
                    # 新規変更: 最新の生フレーム(raw)に検出された枠のみ描画してself.frameに保持する
                    frame_with_boxes = raw.copy()
                    for box in results[0].boxes.data:
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.frame = frame_with_boxes
            if self.detect_count == 0:
                time.sleep(0.1)  # 検出がない場合は少し待機
            else:
                time.sleep(self.cache_duration)  # 通常はキャッシュ持続時間待機

    def record_monitor(self):
        """録画開始の監視を行うメソッド."""
        while self.running:
            with self.lock:
                detect = self.detect_count
                recording = self.recording
                enabled = self.record_enabled
            if enabled and (detect > 0) and (not recording):
                threading.Thread(target=self.record_video, daemon=True).start()
            time.sleep(0.5)

    def record_video(self):
        """録画処理を行うメソッド。検知前15秒と検知終了後15秒も含む映像を保存する."""
        with self.lock:
            if self.recording:
                return
            self.recording = True
            prebuffer = list(self.record_buffer)  # 過去15秒分のフレーム
        # 保存先ディレクトリの作成
        save_dir = os.path.join(self.record_save_path, datetime.now().strftime("%Y%m%d"))
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        full_path = os.path.join(save_dir, file_name)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(full_path, fourcc, self.frame_limit, self.frame_size)
        # 事前バッファの映像を書き込み
        for f in prebuffer:
            writer.write(f)
        off_start = None
        while self.running:
            with self.lock:
                frame = self.raw_frame.copy() if self.raw_frame is not None else None
                current_detect = self.detect_count
            if frame is not None:
                writer.write(frame)
            if current_detect == 0:
                if off_start is None:
                    off_start = time.time()
                elif time.time() - off_start >= 15:
                    break
            else:
                off_start = None
            time.sleep(1 / self.frame_limit)
        writer.release()
        with self.lock:
            self.recording = False

    def get_frame(self, frame_size: tuple = (640, 480), quality: int = 85):
        """
        指定サイズと画質で現在のフレームをJPEG形式にエンコードして返す.
        フレーム未取得時は黒画像に「NoImage」を中央表示する.

        Args:
            frame_size (tuple): 出力画像サイズ.
            quality (int): JPEGエンコード画質.

        return:
            bytes: JPEG形式画像データ.
        """
        with self.lock:
            # 自動的に描画済みの検出結果があればそちらを利用、なければ生フレームを利用
            source_frame = self.frame if self.frame is not None else self.raw_frame
            if source_frame is None:
                # フレーム未取得の場合は黒い画像を生成し、「NoImage」テキストを中央に配置
                frame_to_encode = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
                text = "NoImage"  # 表示テキスト
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = (frame_size[0] - text_size[0]) // 2
                text_y = (frame_size[1] + text_size[1]) // 2
                cv2.putText(frame_to_encode, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            else:
                frame_to_encode = (
                    source_frame if frame_size == self.frame_size else cv2.resize(source_frame, frame_size)
                )
        # 現在時刻をオーバーレイ（映像更新促進のため）
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame_to_encode, current_time, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode(".jpg", frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buffer.tobytes()

    def get_detect_count(self):
        """
        検出オブジェクト数を返す.

        return:
            int: 検出数.
        """
        with self.lock:
            return self.detect_count

    def stop(self):
        """
        フレーム生成および推論スレッドを停止し、カメラリソースを解放する.
        """
        self.running = False
        self.capture_thread.join()
        self.predict_thread.join()
        self.record_monitor_thread.join()
        self.cap.release()


frame_generator = FrameGenerator()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    frame_size = tuple(map(int, request.args.get("frame_size", "640,480").split(",")))
    quality = int(request.args.get("quality", 85))

    def generate():
        while True:
            frame = frame_generator.get_frame(frame_size, quality)
            if frame:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(frame_generator.cache_duration)  # フレーム生成の頻度を制限

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detect_count")
def detect_count():
    count = frame_generator.get_detect_count()
    return jsonify(count=count)


# recordsディレクトリへのパスを設定
RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records")

# ディレクトリが存在しない場合は作成
if not os.path.exists(RECORDS_DIR):
    os.makedirs(RECORDS_DIR)


@app.route("/records")
def records_list():
    """
    recordsフォルダ内のファイル一覧を表示する
    """
    files = []

    # recordsディレクトリ内のファイルを取得
    for filename in os.listdir(RECORDS_DIR):
        file_path = os.path.join(RECORDS_DIR, filename)
        if os.path.isfile(file_path):
            # ファイルの情報を取得
            stats = os.stat(file_path)
            modified_time = datetime.fromtimestamp(stats.st_mtime)

            files.append(
                {
                    "name": filename,
                    "size": humanize.naturalsize(stats.st_size),
                    "modified": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    # 更新日時の新しい順にソート
    files.sort(key=lambda x: x["modified"], reverse=True)

    return render_template("records_list.html", files=files)


@app.route("/download/<filename>")
def download_file(filename):
    """
    ファイルをダウンロードする

    Args:
        filename: ダウンロードするファイル名
    """
    return send_from_directory(RECORDS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    # サーバー設定
    server_config = {"host": "0.0.0.0", "port": 5001, "threads": 8, "backlog": 1024, "channel_timeout": 5}

    # 開発環境判定（環境変数から取得することも検討可能）
    is_development = False

    print(f"AICAMCAM サーバーを起動します（ポート: {server_config['port']}）")

    try:
        if is_development:
            # 開発環境: Flaskの内蔵サーバーを使用
            print("開発モードで起動しています")
            app.run(host=server_config["host"], port=server_config["port"], debug=True)
        else:
            # 本番環境: Waitressサーバーを使用
            print("本番モードで起動しています")
            serve(
                app,
                host=server_config["host"],
                port=server_config["port"],
                threads=server_config["threads"],
                backlog=server_config["backlog"],
                channel_timeout=server_config["channel_timeout"],
            )

    except Exception as e:
        print(f"サーバー起動中にエラーが発生しました: {e}")
    finally:
        print("リソースをクリーンアップしています...")
        frame_generator.stop()
        print("サーバーを終了しました")
