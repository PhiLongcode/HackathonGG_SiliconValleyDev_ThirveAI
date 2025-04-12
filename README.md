## Cấu Trúc Dự Án

```
ThriveAI-AMI-UI/
├── AI2/
│   └── AI2/
│       ├── pycache/
│       ├── static/
│       ├── temp/
│       ├── app_version_3.py
│       ├── app.py
│       ├── dia.json
│       ├── output.mp3
│       ├── output.wav
│       ├── requirements.txt
│       ├── test.py
│       ├── test2.py
│       ├── test3.py
│       └── backend/
│           ├── pycache/
│           ├── main.py
│           └── users.json
├── node_modules/
├── public/
├── src/
├── Test/
├── .env
├── .gitignore
├── bash.exe.stackdump
├── bun.lockb
├── components.json
├── eslint.config.js
├── index.html
├── package-lock.json
├── package.json
├── postcss.config.js
└── README.md
```

---

## Cài Đặt

### Yêu Cầu

- **Node.js** >= 16.x
- **Python** >= 3.8
- `pip`, `virtualenv`

### Hướng Dẫn

1. **Clone dự án:**
   ```bash
   git clone https://github.com/your-repo/thriveai-ami-ui.git
   cd thriveai-ami-ui
   ```

2. **Cài đặt môi trường backend (Python):**
   ```bash
   cd AI2/AI2
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Cài đặt frontend (Node.js):**
   ```bash
   npm install
   ```

---

## Chạy Dự Án

### Backend (FastAPI hoặc Flask)
```bash
python app.py
# hoặc
python app_version_3.py
```

### Frontend (React)
```bash
npm run dev
# hoặc nếu dùng Next.js:
npm run start
```

---

## Tính Năng

- **Nhận diện và phân tích cảm xúc từ âm thanh (MP3/WAV)**
- **Theo dõi tâm trạng hàng ngày**
- **Gợi ý cải thiện sức khỏe tinh thần dựa trên AI**
- **Giao diện tương tác, thân thiện**
- **Tích hợp backend và frontend mượt mà**

---

## Công Nghệ Sử Dụng

### Frontend

- **ReactJS**
- **TailwindCSS**

### Backend

- **Python** (Flask hoặc FastAPI)

### AI/ML

- **Emotion Recognition từ giọng nói**

### Công Cụ Phát Triển

- **Dev Tools**: VSCode, Git, Node.js, Virtualenv
- **API**: Express APIs hoặc FastAPI endpoints

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

- 📧 **Email**: your-email@example.com
- 🌐 **Website**: [https://yourdomain.com](https://yourdomain.com)
- 💬 **Facebook/Zalo**: ThriveAI Team
