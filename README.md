# ThriveAI AMI UI

ThriveAI AMI UI là một ứng dụng hỗ trợ sức khỏe tinh thần, cung cấp các tính năng như theo dõi tâm trạng, gợi ý cải thiện sức khỏe tinh thần, và các công cụ hỗ trợ khác. Dự án bao gồm giao diện người dùng, backend API, và các mô hình AI để phân tích cảm xúc và nhận diện khuôn mặt.

## Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Cài Đặt](#cài-đặt)
- [Chạy Dự Án](#chạy-dự-án)
- [Tính Năng](#tính-năng)
- [Công Nghệ Sử Dụng](#công-nghệ-sử-dụng)
- [Đóng Góp](#đóng-góp)
- [Liên Hệ](#liên-hệ)

## Giới Thiệu

ThriveAI AMI UI là một ứng dụng toàn diện giúp người dùng cải thiện sức khỏe tinh thần thông qua các tính năng như theo dõi tâm trạng, nhật ký cảm xúc, và phân tích cảm xúc bằng AI.


### Yêu Cầu

- **Node.js** >= 16.x
- **Python** >= 3.8
- `pip`, `virtualenv`
- - **Node.js**: v16+
- **npm**: v8+
- **Python**: 3.8+
- **MySQL**: 8.0+ (recommended for database storage)
- **Operating System**: Windows 10+ / macOS Monterey+ / Ubuntu 20.04+
- **RAM**: Minimum 4GB
- **Disk Space**: 2GB

### Cài Đặt

1. **Clone dự án:**
   ```bash
   git clone https://github.com/PhiLongcode/HackathonGG_SiliconValleyDev_ThirveAI
   cd thriveai-ami-ui-main
   ```

2. **Cài đặt môi trường AI (Python):**
   ```bash
   cd AI
   pip install requirements.tx
   
   ```
3. **Cài đặt môi trường Backend (Python):**
   ```bash
   cd bạckend
   pip install fastapi uvicorn
   
   ```
4. **Cài đặt frontend (Node.js):**
   ```bash
   npm install
   ```
---

## Chạy Dự Án
### AI ()
```bash
uvicorn main:app --reload
```
### Backend (FastAPI)
```bash
python main.py
```

### Frontend (React)
```bash
npm run dev
```

---

## Tính Năng
Chức năng đã hoàn thành
- **Nhận diện và phân tích cảm xúc từ âm thanh (MP3/WAV)**
- **Theo dõi tâm trạng hàng ngày**
- **Nhật ký cảm xúc hàng ngày**
- **Gợi ý cải thiện sức khỏe tinh thần dựa trên AI**
- **Giao diện tương tác, thân thiện**
- **Tích hợp backend và frontend mượt mà**

---

## Công Nghệ Sử Dụng

### Frontend

- **ReactJS**
- **TailwindCSS**

### Backend

- **Python** (FastAPI)

### AI/ML

- **Emotion Recognition từ giọng nói**
- **Gemini API**
- **API Reference fast API + uvicorn và web searching API, Installation, Run Locally, Feature**

### Công Cụ Phát Triển

- **Dev Tools**: VSCode, Git, Node.js, Virtualenv
- **API**: FastAPI endpoints

---

## Đóng Góp

Mọi đóng góp đều được hoan nghênh! Để bắt đầu:

1. Fork repository này.
2. Tạo một nhánh mới:
   ```bash
   git checkout -b feature/ten-tinh-nang
   ```
3. Commit thay đổi của bạn:
   ```bash
   git commit -m "Mô tả thay đổi"
   ```
4. Push nhánh của bạn:
   ```bash
   git push origin feature/ten-tinh-nang
   ```
5. Tạo Pull Request (PR) và mô tả thay đổi của bạn.

---

## Liên Hệ
- Email: nguyenphilongls2k4@gmail.com (leader)