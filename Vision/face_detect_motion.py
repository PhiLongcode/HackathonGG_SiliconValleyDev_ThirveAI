import cv2
import numpy as np
import time
import base64
import threading
import queue
import asyncio
from datetime import datetime
from collections import deque
import csv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import random
import os
import sys
import dlib
from deepface import DeepFace

# Đặt chế độ debug
DEBUG = False

# Khai báo biến toàn cục
VERSION = "1.5.0"

# Biến kiểm soát sử dụng thư viện
use_deepface = True
use_landmarks = True
use_fer = False

# Cấu hình facial landmarks
current_dir = os.path.dirname(os.path.abspath(__file__))
shape_predictor_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")

if not os.path.exists(shape_predictor_path):
    raise Exception(f"Không tìm thấy file shape_predictor_68_face_landmarks.dat tại {shape_predictor_path}")

try:
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(shape_predictor_path)
    print("Đã tải thành công bộ phát hiện facial landmarks")
except Exception as e:
    print(f"Không thể sử dụng facial landmarks: {str(e)}")
    use_landmarks = False

# ===== CẤU HÌNH NHẬN DIỆN CẢM XÚC =====

EMOTIONS = [
    "Thư giãn", "Tự hào", "Hạnh phúc", "Nhiệt huyết",
    "Buồn bã", "Sảng khoái", "Cô đơn", "Lo lắng",
    "Hồi hộp", "Tức giận", "Mệt mỏi", "Căng thẳng", "Chán nản"
]

EMOTION_GROUPS = {
    "Tích cực": ["Thư giãn", "Tự hào", "Hạnh phúc", "Nhiệt huyết", "Sảng khoái"],
    "Tiêu cực": ["Buồn bã", "Cô đơn", "Lo lắng", "Hồi hộp", "Tức giận", "Mệt mỏi", "Căng thẳng", "Chán nản"]
}

EMOTION_MAPPING = {
    "angry": ["Tức giận", "Căng thẳng"],
    "disgust": ["Chán nản", "Mệt mỏi"],
    "fear": ["Lo lắng", "Hồi hộp"],
    "happy": ["Hạnh phúc", "Nhiệt huyết", "Sảng khoái"],
    "sad": ["Buồn bã", "Cô đơn"],
    "surprise": ["Hồi hộp", "Nhiệt huyết"],
    "neutral": ["Thư giãn", "Tự hào"]
}

EMOTION_COLORS = {
    "Thư giãn": (170, 232, 128),
    "Tự hào": (147, 112, 219),
    "Hạnh phúc": (0, 215, 255),
    "Nhiệt huyết": (0, 69, 255),
    "Buồn bã": (139, 0, 0),
    "Sảng khoái": (255, 191, 0),
    "Cô đơn": (79, 79, 47),
    "Lo lắng": (255, 144, 30),
    "Hồi hộp": (147, 20, 255),
    "Tức giận": (0, 0, 255),
    "Mệt mỏi": (105, 105, 105),
    "Căng thẳng": (0, 69, 130),
    "Chán nản": (128, 128, 128),
}

DEFAULT_COLOR = (0, 255, 0)

# ===== TIỀN XỬ LÝ DỮ LIỆU =====

def prepare_face_for_emotion(face):
    if face.shape[0] < 48 or face.shape[1] < 48:
        return face
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
    return enhanced

# ===== LỚP CHÍNH: BỘ PHÁT HIỆN CẢM XÚC =====

class FaceEmotionDetector:
    def __init__(self, yolo_model_path="yolo11s.pt"):
        self.start_time = time.time()
        print(f"[{self._get_timestamp()}] Khởi tạo hệ thống nhận diện cảm xúc...")

        self.config = {
            "process_every_n_frames": 2,
            "emotion_stability": 25,
            "confidence_threshold": 0.65,
            "display_fps": True,
            "display_time": True,
            "display_confidence": True,
            "max_faces": 5,
            "skip_similar_faces": True,
            "face_similarity_threshold": 0.6,
            "display_face_landmarks": True,
            "display_emotion_bars": True,
            "rescale_faces": True,
            "face_padding": 0.2,
            "emotion_history_size": 50,
            "smooth_transition": True,
            "confidence_boost": 0.05,
            "emotion_change_threshold": 0.7,
            "use_landmark_rules": True,
            "smoothing_factor": 0.9,
        }

        self.face_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=20)
        self.websocket_active = False
        self.websocket_data = queue.Queue(maxsize=10)
        self.latest_emotions = []  # Lưu trữ kết quả cảm xúc mới nhất

        print(f"[{self._get_timestamp()}] Đang tải mô hình YOLO...")
        self.yolo_model = YOLO(yolo_model_path)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Không thể mở camera")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[{self._get_timestamp()}] Camera: {self.frame_width}x{self.frame_height}@{self.fps}fps")

        self.frame_count = 0
        self.processed_frames = 0
        self.face_emotions = {}
        self.emotion_history = {}
        self.emotion_confidence = {}
        self.face_positions = {}
        self.face_timestamps = {}
        self.face_landmarks = {}
        self.emotion_scores = {}
        self.emotion_window = {}

        self.csv_file = open('emotion_log.csv', 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'Face_ID', 'Emotion', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

        self.running = True
        self.processor_thread = threading.Thread(target=self._process_faces)
        self.processor_thread.daemon = True
        self.processor_thread.start()

        print(f"[{self._get_timestamp()}] Hệ thống đã sẵn sàng!")

    def _get_timestamp(self):
        return datetime.now().strftime("%H:%M:%S")

    def _track_face(self, face_id, face_rect):
        current_time = time.time()
        if face_id in self.face_positions:
            prev_rect = self.face_positions[face_id]
            alpha = self.config["smoothing_factor"]
            smoothed_rect = (
                int(alpha * prev_rect[0] + (1 - alpha) * face_rect[0]),
                int(alpha * prev_rect[1] + (1 - alpha) * face_rect[1]),
                int(alpha * prev_rect[2] + (1 - alpha) * face_rect[2]),
                int(alpha * prev_rect[3] + (1 - alpha) * face_rect[3])
            )
            self.face_positions[face_id] = smoothed_rect
        else:
            self.face_positions[face_id] = face_rect

        self.face_timestamps[face_id] = current_time
        if face_id not in self.emotion_window:
            self.emotion_window[face_id] = {}
            for emotion in EMOTIONS:
                self.emotion_window[face_id][emotion] = deque([0.0] * 10, maxlen=10)

        expired_ids = []
        for old_id, timestamp in self.face_timestamps.items():
            if current_time - timestamp > 3.0:
                expired_ids.append(old_id)

        for expired_id in expired_ids:
            if expired_id in self.face_timestamps:
                del self.face_timestamps[expired_id]
            if expired_id in self.face_positions:
                del self.face_positions[expired_id]
            if expired_id in self.face_emotions:
                del self.face_emotions[expired_id]
            if expired_id in self.emotion_history:
                del self.emotion_history[expired_id]
            if expired_id in self.emotion_confidence:
                del self.emotion_confidence[expired_id]
            if expired_id in self.face_landmarks:
                del self.face_landmarks[expired_id]
            if expired_id in self.emotion_scores:
                del self.emotion_scores[expired_id]
            if expired_id in self.emotion_window:
                del self.emotion_window[expired_id]

        return self.face_positions[face_id]

    def _calculate_face_similarity(self, rect1, rect2):
        center1 = ((rect1[0] + rect1[2]) // 2, (rect1[1] + rect1[3]) // 2)
        center2 = ((rect2[0] + rect2[2]) // 2, (rect2[1] + rect2[3]) // 2)
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        size1 = (rect1[2] - rect1[0] + rect1[3] - rect1[1]) / 2
        size2 = (rect2[2] - rect2[0] + rect2[3] - rect2[1]) / 2
        avg_size = (size1 + size2) / 2
        similarity = max(0, 1 - distance / (avg_size * 2))
        return similarity

    def _find_matching_face_id(self, face_rect):
        best_match = None
        best_score = -1
        for face_id, existing_rect in self.face_positions.items():
            similarity = self._calculate_face_similarity(face_rect, existing_rect)
            if similarity > self.config["face_similarity_threshold"] and similarity > best_score:
                best_match = face_id
                best_score = similarity
        return best_match

    def _generate_face_id(self):
        return f"face_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

    def _update_emotion_history(self, face_id, emotion, confidence):
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = []
        self.emotion_history[face_id].append((emotion, confidence))
        if len(self.emotion_history[face_id]) > self.config["emotion_history_size"]:
            self.emotion_history[face_id].pop(0)
        if face_id not in self.emotion_confidence:
            self.emotion_confidence[face_id] = {}
        self.emotion_window[face_id][emotion].append(confidence)
        if emotion in self.emotion_confidence[face_id]:
            self.emotion_confidence[face_id][emotion] += confidence * self.config["confidence_boost"]
        else:
            self.emotion_confidence[face_id][emotion] = confidence
        for other_emotion in self.emotion_confidence[face_id]:
            if other_emotion != emotion:
                self.emotion_confidence[face_id][other_emotion] *= 0.92
        for emotion_name in list(self.emotion_confidence[face_id].keys()):
            if self.emotion_confidence[face_id][emotion_name] < 0.05:
                del self.emotion_confidence[face_id][emotion_name]

    def _smooth_emotion_scores(self, face_id):
        if face_id not in self.emotion_window:
            return {}
        smoothed_scores = {}
        for emotion in EMOTIONS:
            if emotion in self.emotion_window[face_id]:
                scores = list(self.emotion_window[face_id][emotion])
                if scores:
                    weights = [0.1, 0.1, 0.2, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0][-len(scores):]
                    weights = [w / sum(weights) for w in weights]
                    smoothed_scores[emotion] = sum(s * w for s, w in zip(scores, weights))
        return smoothed_scores

    def _get_stable_emotion(self, face_id):
        if face_id not in self.emotion_history or len(self.emotion_history[face_id]) < 3:
            return None, 0
        current_emotion = self.face_emotions.get(face_id)
        smoothed_scores = self._smooth_emotion_scores(face_id)
        emotion_counts = {}
        emotion_confidences = {}
        max_history = min(20, len(self.emotion_history[face_id]))
        recent_history = self.emotion_history[face_id][-max_history:]
        for emotion, confidence in recent_history:
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
                emotion_confidences[emotion] += confidence
            else:
                emotion_counts[emotion] = 1
                emotion_confidences[emotion] = confidence
        for emotion in emotion_confidences:
            emotion_confidences[emotion] /= emotion_counts[emotion]
        sorted_emotions = sorted(
            emotion_counts.items(),
            key=lambda x: (x[1], emotion_confidences.get(x[0], 0), smoothed_scores.get(x[0], 0)),
            reverse=True
        )
        if (current_emotion and
                current_emotion in [e[0] for e in sorted_emotions[:2]] and
                emotion_counts.get(current_emotion, 0) >= self.config["emotion_stability"] // 3):
            confidence = max(
                emotion_confidences.get(current_emotion, 0.5),
                smoothed_scores.get(current_emotion, 0.5)
            )
            return current_emotion, confidence
        if sorted_emotions and sorted_emotions[0][1] >= self.config["emotion_stability"] // 4:
            top_emotion = sorted_emotions[0][0]
            confidence = max(
                emotion_confidences.get(top_emotion, 0.5),
                smoothed_scores.get(top_emotion, 0.5)
            )
            if current_emotion and confidence < self.config["emotion_change_threshold"]:
                return current_emotion, emotion_confidences.get(current_emotion, 0.5)
            return top_emotion, confidence
        return current_emotion, emotion_confidences.get(current_emotion, 0.5) if current_emotion else (None, 0)

    def _analyze_facial_landmarks(self, landmarks):
        if not landmarks or len(landmarks) != 68:
            return {}
        points = np.array(landmarks)
        left_eye = points[36:42]
        right_eye = points[42:48]
        mouth = points[48:68]
        left_eyebrow = points[17:22]
        right_eyebrow = points[22:27]
        nose = points[27:36]
        jaw = points[0:17]
        left_eye_h = np.linalg.norm(left_eye[1] - left_eye[5])
        left_eye_w = np.linalg.norm(left_eye[0] - left_eye[3])
        left_eye_ratio = left_eye_h / left_eye_w if left_eye_w > 0 else 0
        right_eye_h = np.linalg.norm(right_eye[1] - right_eye[5])
        right_eye_w = np.linalg.norm(right_eye[0] - right_eye[3])
        right_eye_ratio = right_eye_h / right_eye_w if right_eye_w > 0 else 0
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        mouth_h = np.linalg.norm(mouth[3] - mouth[9])
        mouth_w = np.linalg.norm(mouth[0] - mouth[6])
        mouth_ratio = mouth_h / mouth_w if mouth_w > 0 else 0
        mouth_curve = (mouth[3][1] + mouth[9][1]) / 2 - (mouth[0][1] + mouth[6][1]) / 2
        left_brow_h = np.mean([p[1] for p in left_eyebrow])
        right_brow_h = np.mean([p[1] for p in right_eyebrow])
        brow_height = (left_brow_h + right_brow_h) / 2
        emotion_scores = {}
        if mouth_curve < 0 and mouth_ratio > 0.3:
            emotion_scores["Hạnh phúc"] = min(1.0, 0.5 + abs(mouth_curve) * 5)
            emotion_scores["Sảng khoái"] = min(1.0, 0.4 + abs(mouth_curve) * 3)
            emotion_scores["Nhiệt huyết"] = min(1.0, 0.3 + abs(mouth_curve) * 2)
        if mouth_curve > 0 and eye_ratio < 0.3:
            emotion_scores["Buồn bã"] = min(1.0, 0.5 + mouth_curve * 3)
            emotion_scores["Cô đơn"] = min(1.0, 0.4 + mouth_curve * 2)
            emotion_scores["Chán nản"] = min(1.0, 0.3 + mouth_curve * 2)
        if abs(mouth_curve) < 0.1 and brow_height < 0.4:
            emotion_scores["Tức giận"] = min(1.0, 0.6 - brow_height)
            emotion_scores["Căng thẳng"] = min(1.0, 0.5 - brow_height)
        if eye_ratio > 0.35 and brow_height < 0.3:
            emotion_scores["Lo lắng"] = min(1.0, 0.4 + eye_ratio)
            emotion_scores["Hồi hộp"] = min(1.0, 0.3 + eye_ratio)
        if eye_ratio < 0.25 and mouth_ratio < 0.2:
            emotion_scores["Mệt mỏi"] = min(1.0, 0.8 - eye_ratio)
            emotion_scores["Thư giãn"] = min(1.0, 0.6 - eye_ratio)
        if 0.25 < eye_ratio < 0.35 and abs(mouth_curve) < 0.05:
            emotion_scores["Thư giãn"] = 0.8
            emotion_scores["Tự hào"] = 0.7
        return emotion_scores

    def _extract_landmarks(self, face_img, face_rect):
        if not use_landmarks:
            return None
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            x1, y1, x2, y2 = face_rect
            dlib_rect = dlib.rectangle(x1, y1, x2, y2)
            shape = landmark_predictor(gray, dlib_rect)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            return landmarks
        except Exception as e:
            if DEBUG:
                print(f"Lỗi khi trích xuất điểm đặc trưng: {str(e)}")
            return None

    def map_emotion(self, emotion_scores, face_id=None, face_landmarks=None, source="deepface"):
        if isinstance(emotion_scores, dict):
            top_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            confidence = emotion_scores[top_emotion]
        else:
            top_emotion = emotion_scores
            confidence = 0.7
        landmark_scores = {}
        if face_landmarks and self.config["use_landmark_rules"] and face_id:
            if face_id in self.face_landmarks:
                landmark_scores = self._analyze_facial_landmarks(self.face_landmarks[face_id])
                if face_id not in self.emotion_scores:
                    self.emotion_scores[face_id] = {}
                for emotion, score in landmark_scores.items():
                    if emotion in self.emotion_scores[face_id]:
                        alpha = 0.7
                        self.emotion_scores[face_id][emotion] = alpha * self.emotion_scores[face_id][emotion] + (1 - alpha) * score
                    else:
                        self.emotion_scores[face_id][emotion] = score
        if top_emotion in EMOTION_MAPPING:
            options = EMOTION_MAPPING[top_emotion]
            if landmark_scores and face_id in self.emotion_scores:
                best_emotion = None
                best_score = -1
                for emotion in options:
                    if emotion in self.emotion_scores[face_id]:
                        if self.emotion_scores[face_id][emotion] > best_score:
                            best_score = self.emotion_scores[face_id][emotion]
                            best_emotion = emotion
                if best_emotion and best_score > 0.5:
                    return best_emotion, max(confidence, best_score)
            if confidence > 0.8:
                weight = [3, 2, 1]
            else:
                weight = [1, 1, 1]
            weights = weight[:len(options)] if len(weight) >= len(options) else weight + [1] * (len(options) - len(weight))
            if options:
                emotion = random.choices(options, weights=weights[:len(options)], k=1)[0]
                return emotion, confidence
        if landmark_scores and face_id in self.emotion_scores:
            best_emotion = None
            best_score = 0.4
            for emotion, score in self.emotion_scores[face_id].items():
                if score > best_score:
                    best_score = score
                    best_emotion = emotion
            if best_emotion:
                return best_emotion, best_score
        emotion = random.choice(EMOTIONS)
        return emotion, 0.5

    def detect_faces_yolo(self, frame):
        results = self.yolo_model(
            frame,
            classes=[0],
            conf=self.config["confidence_threshold"]
        )
        faces = []
        if results[0].boxes:
            boxes = results[0].boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                face_y2 = y1 + int((y2 - y1) * 0.4)
                person_upper = frame[y1:face_y2, x1:x2]
                if person_upper.size == 0 or person_upper.shape[0] < 40 or person_upper.shape[1] < 40:
                    continue
                try:
                    gray_upper = cv2.cvtColor(person_upper, cv2.COLOR_BGR2GRAY)
                    detected_faces = self.face_cascade.detectMultiScale(
                        gray_upper, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                except Exception as e:
                    if DEBUG:
                        print(f"Lỗi phát hiện khuôn mặt: {str(e)}")
                    detected_faces = []
                for (fx, fy, fw, fh) in detected_faces:
                    face_x1 = x1 + fx
                    face_y1 = y1 + fy
                    face_x2 = face_x1 + fw
                    face_y2 = face_y1 + fh
                    if self.config["face_padding"] > 0:
                        padding_x = int(fw * self.config["face_padding"])
                        padding_y = int(fh * self.config["face_padding"])
                        face_x1 = max(0, face_x1 - padding_x)
                        face_y1 = max(0, face_y1 - padding_y)
                        face_x2 = min(frame.shape[1], face_x2 + padding_x)
                        face_y2 = min(frame.shape[0], face_y2 + padding_y)
                    face = frame[face_y1:face_y2, face_x1:face_x2]
                    if face.size > 0 and face.shape[0] >= 30 and face.shape[1] >= 30:
                        face_rect = (face_x1, face_y1, face_x2, face_y2)
                        face_id = self._find_matching_face_id(face_rect)
                        if face_id is None:
                            face_id = self._generate_face_id()
                        tracked_rect = self._track_face(face_id, face_rect)
                        prepared_face = prepare_face_for_emotion(face)
                        landmarks = self._extract_landmarks(frame, tracked_rect)
                        if landmarks:
                            self.face_landmarks[face_id] = landmarks
                        faces.append({
                            "id": face_id,
                            "face": prepared_face,
                            "coords": tracked_rect,
                            "person_coords": (x1, y1, x2, y2),
                            "confidence": confidence,
                            "landmarks": landmarks,
                            "original_frame": frame,
                        })
                        if len(faces) >= self.config["max_faces"]:
                            break
        return faces

    def detect_faces_opencv(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = []
        for (x, y, w, h) in detected_faces:
            if self.config["face_padding"] > 0:
                padding_x = int(w * self.config["face_padding"])
                padding_y = int(h * self.config["face_padding"])
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(frame.shape[1], x + w + padding_x)
                y2 = min(frame.shape[0], y + h + padding_y)
            else:
                x1, y1, x2, y2 = x, y, x + w, y + h
            face = frame[y1:y2, x1:x2]
            if face.size > 0 and face.shape[0] >= 30 and face.shape[1] >= 30:
                face_rect = (x1, y1, x2, y2)
                face_id = self._find_matching_face_id(face_rect)
                if face_id is None:
                    face_id = self._generate_face_id()
                tracked_rect = self._track_face(face_id, face_rect)
                prepared_face = prepare_face_for_emotion(face)
                landmarks = self._extract_landmarks(frame, tracked_rect)
                if landmarks:
                    self.face_landmarks[face_id] = landmarks
                faces.append({
                    "id": face_id,
                    "face": prepared_face,
                    "coords": tracked_rect,
                    "person_coords": None,
                    "confidence": 0.7,
                    "landmarks": landmarks,
                    "original_frame": frame,
                })
                if len(faces) >= self.config["max_faces"]:
                    break
        return faces

    def analyze_emotions(self, face_data):
        face = face_data["face"]
        face_id = face_data["id"]
        landmarks = face_data["landmarks"]
        try:
            if use_deepface:
                analysis = DeepFace.analyze(
                    face, actions=['emotion'], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]
                emotion, confidence = self.map_emotion(analysis['emotion'], face_id, landmarks, "deepface")
            elif landmarks and self.config["use_landmark_rules"]:
                landmark_scores = self._analyze_facial_landmarks(landmarks)
                if landmark_scores:
                    top_emotion = max(landmark_scores.items(), key=lambda x: x[1])[0]
                    confidence = landmark_scores[top_emotion]
                    emotion = top_emotion
            else:
                hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
                h_mean = np.mean(hsv[:, :, 0])
                s_mean = np.mean(hsv[:, :, 1])
                v_mean = np.mean(hsv[:, :, 2])
                if v_mean < 100:
                    emotions_pool = ["Buồn bã", "Chán nản", "Cô đơn", "Mệt mỏi"]
                elif s_mean > 100:
                    emotions_pool = ["Hạnh phúc", "Tức giận", "Nhiệt huyết", "Sảng khoái"]
                elif h_mean < 30 or h_mean > 150:
                    emotions_pool = ["Tức giận", "Lo lắng", "Căng thẳng", "Hồi hộp"]
                else:
                    emotions_pool = ["Thư giãn", "Tự hào", "Hạnh phúc", "Sảng khoái"]
                emotion = random.choice(emotions_pool) if face_id not in self.face_emotions else self.face_emotions[face_id]
                confidence = random.uniform(0.6, 0.9)
            self._update_emotion_history(face_id, emotion, confidence)
            stable_emotion, stable_confidence = self._get_stable_emotion(face_id)
            if stable_emotion:
                self.face_emotions[face_id] = stable_emotion
                emotion = stable_emotion
                confidence = stable_confidence
            x1, y1, x2, y2 = face_data["coords"]
            timestamp = self._get_timestamp()
            self.csv_writer.writerow([timestamp, face_id, emotion, f"{confidence:.2f}", x1, y1, x2, y2])
            self.csv_file.flush()
            return {
                "id": face_id,
                "emotion": emotion,
                "confidence": confidence,
                "coords": face_data["coords"],
                "person_coords": face_data["person_coords"],
                "landmarks": landmarks,
            }
        except Exception as e:
            if DEBUG:
                print(f"Lỗi phân tích cảm xúc: {str(e)}")
            emotion = self.face_emotions.get(face_id, random.choice(EMOTIONS))
            confidence = 0.5
            x1, y1, x2, y2 = face_data["coords"]
            timestamp = self._get_timestamp()
            self.csv_writer.writerow([timestamp, face_id, emotion, f"{confidence:.2f}", x1, y1, x2, y2])
            self.csv_file.flush()
            return {
                "id": face_id,
                "emotion": emotion,
                "confidence": confidence,
                "coords": face_data["coords"],
                "person_coords": face_data["person_coords"],
                "landmarks": landmarks,
            }

    def _process_faces(self):
        while self.running:
            try:
                if not self.face_queue.empty():
                    face_data = self.face_queue.get(timeout=0.1)
                    result = self.analyze_emotions(face_data)
                    self.result_queue.put(result)
                    self.face_queue.task_done()
                else:
                    time.sleep(0.01)
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                if DEBUG:
                    print(f"Lỗi xử lý khuôn mặt: {str(e)}")
                time.sleep(0.05)

    def _draw_emotion_bars(self, frame, face_id, emotion, x1, y1, x2, y2):
        if not self.config["display_emotion_bars"] or face_id not in self.emotion_confidence:
            return
        confidences = self.emotion_confidence[face_id]
        filtered_confidences = {e: c for e, c in confidences.items() if c > 0.1}
        if not filtered_confidences:
            return
        total = sum(filtered_confidences.values())
        if total <= 0:
            return
        bar_height = 4
        bar_width = x2 - x1
        bar_top = y2 + 5
        bar_gap = 2
        sorted_emotions = sorted(filtered_confidences.items(), key=lambda x: x[1], reverse=True)
        top_emotions = sorted_emotions[:3]
        for i, (emo, conf) in enumerate(top_emotions):
            bar_y = bar_top + i * (bar_height + bar_gap)
            normalized_conf = conf / total
            bar_length = int(bar_width * normalized_conf)
            color = EMOTION_COLORS.get(emo, DEFAULT_COLOR)
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_height), (50, 50, 50), -1)
            cv2.rectangle(frame, (x1, bar_y), (x1 + bar_length, bar_y + bar_height), color, -1)
            text_y = bar_y + bar_height + 10
            cv2.putText(frame, f"{emo}", (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def _draw_facial_landmarks(self, frame, landmarks, color=(0, 255, 0)):
        if not landmarks:
            return
        for point in landmarks:
            cv2.circle(frame, (int(point[0]), int(point[1])), 1, color, -1)
        for i in range(16):
            p1 = landmarks[i]
            p2 = landmarks[i + 1]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)
        for i in range(17, 21):
            p1 = landmarks[i]
            p2 = landmarks[i + 1]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)
        for i in range(22, 26):
            p1 = landmarks[i]
            p2 = landmarks[i + 1]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)
        for i in range(27, 30):
            p1 = landmarks[i]
            p2 = landmarks[i + 1]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)
        for i in range(36, 41):
            p1 = landmarks[i]
            p2 = landmarks[i + 1]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)
        cv2.line(frame, (int(landmarks[41][0]), int(landmarks[41][1])),
                 (int(landmarks[36][0]), int(landmarks[36][1])), color, 1)
        for i in range(42, 47):
            p1 = landmarks[i]
            p2 = landmarks[i + 1]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)
        cv2.line(frame, (int(landmarks[47][0]), int(landmarks[47][1])),
                 (int(landmarks[42][0]), int(landmarks[42][1])), color, 1)
        for i in range(48, 59):
            p1 = landmarks[i]
            p2 = landmarks[i + 1]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)
        cv2.line(frame, (int(landmarks[59][0]), int(landmarks[59][1])),
                 (int(landmarks[48][0]), int(landmarks[48][1])), color, 1)
        for i in range(60, 67):
            p1 = landmarks[i]
            p2 = landmarks[i + 1]
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)
        cv2.line(frame, (int(landmarks[67][0]), int(landmarks[67][1])),
                 (int(landmarks[60][0]), int(landmarks[60][1])), color, 1)

    async def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame_count += 1
        process_this_frame = (self.frame_count % self.config["process_every_n_frames"] == 0)
        display_frame = frame.copy()
        results = []
        if process_this_frame:
            self.processed_frames += 1
            faces = self.detect_faces_yolo(frame)
            if len(faces) == 0:
                faces = self.detect_faces_opencv(frame)
            for face_data in faces:
                try:
                    self.face_queue.put(face_data, timeout=0.03)
                except queue.Full:
                    pass
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get(timeout=0.01)
                results.append(result)
                self.result_queue.task_done()
            except queue.Empty:
                break
        for result in results:
            face_id = result["id"]
            emotion = result["emotion"]
            confidence = result["confidence"]
            (x1, y1, x2, y2) = result["coords"]
            landmarks = result["landmarks"]
            color = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            text = f"{emotion}"
            if self.config["display_confidence"] and confidence > 0.6:
                text += f" ({int(confidence * 100)}%)"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.rectangle(display_frame,
                          (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5),
                          color, -1)
            cv2.putText(display_frame, text,
                        (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            if landmarks and self.config["display_face_landmarks"]:
                self._draw_facial_landmarks(display_frame, landmarks, color)
            self._draw_emotion_bars(display_frame, face_id, emotion, x1, y2 + 5, x2, y2 + 50)
        _, buffer = cv2.imencode('.jpg', display_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        # Lưu kết quả cảm xúc mới nhất
        self.latest_emotions = [
            {
                "face_id": r["id"],
                "emotion": r["emotion"],
                "confidence": f"{r['confidence']:.2f}",
                "coords": [int(x) for x in r["coords"]]
            } for r in results
        ]
        return {
            "frame": frame_base64,
            "results": self.latest_emotions
        }

    def get_latest_emotions(self):
        return self.latest_emotions

    def cleanup(self):
        self.running = False
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.cleanup()

# ===== FASTAPI SERVER =====

app = FastAPI(title="Emotion Detection API")
detector = None

@app.on_event("startup")
async def startup_event():
    global detector
    model_path = "yolo11s.pt"
    if not os.path.exists(model_path):
        raise Exception(f"Không tìm thấy file mô hình YOLO '{model_path}'")
    detector = FaceEmotionDetector(model_path)
    print("API server đã khởi động và detector đã sẵn sàng.")

@app.on_event("shutdown")
async def shutdown_event():
    global detector
    if detector:
        detector.cleanup()
        detector = None
    print("API server đã dừng.")

@app.get("/emotions")
async def get_emotions():
    global detector
    if not detector:
        raise HTTPException(status_code=503, detail="Detector chưa được khởi tạo")
    # Xử lý một khung hình mới để cập nhật cảm xúc
    await detector.process_frame()
    emotions = detector.get_latest_emotions()
    return {"results": emotions}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print(f"HỆ THỐNG NHẬN DIỆN CẢM XÚC KHUÔN MẶT v{VERSION} với FastAPI")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000)