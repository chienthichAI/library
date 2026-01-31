# SmartLib Kiosk - AI-Powered Library System

## 🚀 Quick Start

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn app.main:app --reload
```

### API Documentation

After starting the server, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📦 Project Structure

```
thuvien/
├── backend/
│   ├── app/
│   │   ├── api/routes/      # FastAPI endpoints
│   │   ├── ml/              # AI/ML modules
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic
│   │   ├── config.py        # Configuration
│   │   ├── database.py      # Database connection
│   │   └── main.py          # FastAPI app
│   ├── requirements.txt
│   └── run.bat
├── frontend/                 # React (TODO)
└── README.md
```

## 🔐 API Endpoints

### Authentication
- `POST /api/v1/auth/verify-face` - Verify student identity via face
- `POST /api/v1/auth/register-face` - Register student face

### Books
- `POST /api/v1/books/detect` - Detect book from image
- `GET /api/v1/books/{barcode}` - Get book info
- `GET /api/v1/books/` - List all books
- `POST /api/v1/books/` - Create new book

### Transactions
- `POST /api/v1/transactions/borrow` - Borrow a book
- `POST /api/v1/transactions/return` - Return a book
- `GET /api/v1/transactions/history/{student_id}` - Get history

### Students
- `GET /api/v1/students/{id}` - Get student info
- `GET /api/v1/students/{id}/borrowing-info` - Get borrowing status
- `POST /api/v1/students/` - Create student

## 🤖 AI/ML Models

| Model | Purpose | Architecture |
|-------|---------|--------------|
| **ArcFace** | Face Recognition | ResNet100, 512-dim embedding |
| **MiniFASNet** | Anti-Spoofing | MobileFaceNet |
| **YOLOv8** | Book Detection | CSPDarknet + PANet |
| **PaddleOCR** | Text Extraction | DB + CRNN |

## 🗄️ Database (Supabase)

Tables:
- `students` - Student information
- `face_embeddings` - Face vectors (512-dim)
- `books` - Library catalog
- `transactions` - Borrow/return records

## 📝 Environment Variables

Copy `.env.example` to `.env` and configure:

```env
DATABASE_URL=postgresql+asyncpg://...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key
```

## 🧪 Testing

```bash
cd backend
pytest tests/ -v
```

## 📄 License

MIT License
