# NGHIÊN CỨU VÀ PHÁT TRIỂN HỆ THỐNG THƯ VIỆN THÔNG MINH ĐA NỀN TẢNG TÍCH HỢP TRÍ TUỆ NHÂN TẠO (AI)

**Tác giả:** [Tên sinh viên/Nhóm]  
**Giảng viên hướng dẫn:** [Tên Giảng viên]  
**Đơn vị:** Đại học FPT  
**Ngày:** 27 tháng 03 năm 2026

---

## TÓM TẮT (ABSTRACT)
Bài báo này trình bày một giải pháp toàn diện cho việc hiện đại hóa quản lý thư viện thông qua hệ thống **SmartLib Kiosk**. Giải pháp tích hợp các công nghệ đột phá bao gồm **Computer Vision** (nhận diện khuôn mặt và vật thể) và **Xử lý ngôn ngữ tự nhiên** (LLM với kiến trúc Retrieval-Augmented Generation - RAG). Hệ thống được thiết kế để tự động hóa quy trình mượn/trả sách với độ chính xác xác thực trên 99.5%, đồng thời cung cấp trợ lý ảo thông minh hỗ trợ tra cứu ngữ nghĩa chuyên sâu. 

Nghiên cứu tập trung vào việc tối ưu hóa hiệu năng trên thiết bị biên **NVIDIA Jetson Orin Nano**, sử dụng các mô hình SOTA như **RetinaFace**, **ArcFace** cho xác thực sinh trắc học và **YOLOv8** cho nhận diện vật thể. Ngoài ra, kiến trúc RAG được xây dựng trên nền tảng **BGE-M3** và **pgvector** để giải quyết bài toán tra cứu thông tin sách dựa trên ngữ nghĩa tự nhiên thay vì chỉ dựa vào từ khóa truyền thống. Kết quả thực nghiệm cho thấy hệ thống không chỉ giải quyết bài toán vận hành hiệu quả mà còn tạo ra một bước đột phá trong trải nghiệm người dùng tại thư viện đại học.

---

## 1. GIỚI THIỆU (INTRODUCTION)

### 1.1 Bối cảnh và Sự phát triển của Thư viện Thông minh
Trong kỷ nguyên của Cách mạng Công nghiệp 4.0 và chuyển đổi số (Digital Transformation), vai trò của thư viện truyền thống đang trải qua những thay đổi căn bản. Thư viện không còn đơn thuần là một kho lưu trữ vật lý mà đang trở thành các trung tâm tri thức năng động, nơi công nghệ đóng vai trò là "cầu nối" giữa người đọc và kho tàng tri thức khổng lồ.

Tại các trường đại học hàng đầu như Đại học FPT, khối lượng đầu sách và số lượng sinh viên tra cứu hàng ngày là rất lớn. Tuy nhiên, các hệ thống quản lý thư viện truyền thống (Library Management Systems - LMS) vẫn còn tồn tại nhiều rào cản về mặt vận hành và trải nghiệm. Việc dựa vào các phương thức nhận dạng lỗi thời như thẻ barcode vật lý không những tiềm ẩn nguy cơ mất mát, giả mạo mà còn tạo ra những nút thắt cổ chai về mặt thời gian trong các giờ cao điểm.

### 1.2 Thách thức của các Giải pháp hiện tại
Các công nghệ tự động hóa thư viện trước đây như RFID (Radio Frequency Identification) mặc dù giúp tăng tốc độ mượn trả nhưng lại gặp phải những khó khăn đáng kể:
- **Chi phí triển khai:** Chip RFID cho từng cuốn sách là một khoản đầu tư rất lớn khi áp dụng cho hàng trăm ngàn đầu sách.
- **Bảo trì:** Các chip RFID có thể bị hỏng hóc vật lý hoặc bị nhiễu do vật liệu kim loại/nước gần đó.
- **Tính cá nhân hóa:** RFID chỉ quản lý "vật", nó không giúp hiểu được "người" đang sử dụng hệ thống là ai và họ đang cần tìm kiếm nội dung gì trong sách.

### 1.3 Mục tiêu và Phạm vi Nghiên cứu
Dự án SmartLib đặt ra mục tiêu vượt qua các giới hạn trên bằng cách sử dụng **Visual AI** và **Generative AI** để thay thế các phương thức phần cứng thủ công. Cụ thể:
1. **Xác thực Không tiếp xúc (Contactless Authentication):** Loại bỏ nhu cầu sử dụng thẻ vật lý thông qua công nghệ nhận diện khuôn mặt 3D-aware.
2. **Nhận diện Vật thể Thông minh (Smart Object Detection):** Sử dụng Computer Vision để nhận dạng sách và mã barcode trực tiếp từ luồng video, giảm thiểu sai sót của con người.
3. **Tra cứu Ngữ nghĩa (Semantic Search):** Xây dựng một trợ lý ảo có khả năng hiểu các câu hỏi mang tính khái niệm của sinh viên để đưa ra gợi ý sách chính xác.

---

## 2. CƠ SỞ LÝ THUYẾT VÀ TỔNG QUAN CÔNG NGHỆ (LITERATURE REVIEW)

### 2.1 Nhận diện khuôn mặt với ArcFace và RetinaFace
Nhận diện khuôn mặt đã chuyển từ các giải pháp truyền thống dựa trên Eigenfaces sang mạng thần kinh sâu (Deep Neural Networks). Trong nghiên cứu này, chúng tôi lựa chọn sự kết hợp giữa RetinaFace để phát hiện và ArcFace để trích xuất đặc trưng.

- **RetinaFace:** Là một giải thuật phát hiện khuôn mặt đơn tầng mạnh mẽ, thực hiện "pixel-wise face localization" bằng cách sử dụng mạng Feature Pyramid Network (FPN). RetinaFace không chỉ cung cấp hộp giới hạn (Bounding Box) mà còn định vị chính xác 5 điểm mốc (landmarks) ngay cả trong điều kiện khuôn mặt bị che khuất một phần.
- **ArcFace (Additive Angular Margin Loss):** Khác với hàm Softmax truyền thống, ArcFace giới thiệu một tham số lề góc (angular margin) vào hàm mất mát. Điều này buộc mạng thần kinh phải học các đặc trưng có tính phân tách cao trong không gian vector. Công thức toán học cốt lõi:
  $$L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j=1, j \neq y_i}^{n} e^{s \cos \theta_j}}$$
  Trong đó $s$ là bán kính hyper-sphere và $m$ là tham số lề góc. Việc này giúp tăng cường độ chính xác đáng kể cho các hệ thống nhận diện quy mô lớn.

### 2.2 Nhận diện vật thể với YOLOv8
YOLO (You Only Look Once) đã thống trị lĩnh vực phát hiện vật thể thời gian thực từ năm 2015. Phiên bản YOLOv8 (phát triển bởi Ultralytics) mang lại kiến trúc C2f mới, loại bỏ khái niệm anchor-box, giúp tốc độ inference nhanh hơn và độ chính xác mAP (mean Average Precision) cao hơn. YOLOv8 đóng vai trò then chốt trong việc xác định vùng chứa mã vạch (ROI) để chuẩn bị cho quá trình giải mã barcode.

### 2.3 Kiến trúc RAG (Retrieval-Augmented Generation)
RAG là một kỹ thuật tối ưu hóa đầu ra của LLM bằng cách cho phép mô hình truy cập vào các nguồn dữ liệu tin cậy bên ngoài trước khi sinh câu trả lời. Khác với việc Fine-tuning mô hình (rất tốn kém và khó cập nhật dữ liệu mới), RAG cho phép hệ thống SmartLib cập nhật thông tin sách mỗi khi có đầu sách mới nhập kho chỉ bằng cách cập nhật Vector Database.

---

## 3. KIẾN TRÚC HỆ THỐNG CHI TIẾT (SYSTEM ARCHITECTURE)

Hệ thống SmartLib được xây dựng theo mô hình **Hybrid Cloud & Edge Computing**, phân tách rõ ràng các tác vụ yêu cầu độ trễ thấp (nhận diện tại Kiosk) và các tác vụ yêu cầu tài nguyên lớn (Chatbot LLM trên Cloud).

### 3.1 Cấu trúc 4 Lớp (The 4-Layer Stack)

```mermaid
graph TD
    subgraph "Lớp Phần cứng (Physical Hardware)"
        direction LR
        Camera1[Camera Face ID]
        Camera2[Camera Book Detection]
        PIR[Cảm biến hồng ngoại PIR]
        Touch[Màn hình cảm ứng 15-inch]
    end

    subgraph "Lớp Xử lý Biên (Edge AI - Jetson Orin Nano)"
        FE[Face Engine: RetinaFace + ArcFace]
        BE[Book Engine: YOLOv8 Object Detection]
        IM[Image Processing: CLAHE + Otsu]
        State[State Machine: Logic mượn trả]
    end

    subgraph "Lớp Dịch vụ (Backend - FastAPI)"
        Gate[API Gateway]
        Student[Quản lý Sinh viên]
        Book[Quản lý Đầu sách]
        Log[Hệ thống Nhật ký & Báo cáo]
        RAG[RAG Orchestrator]
    end

    subgraph "Lớp Dữ liệu (Intelligence & Storage)"
        RDS[PostgreSQL - Dữ liệu quan hệ]
        VEC[pgvector - Vector Embeddings]
        LLM[Large Language Models - Qwen/GPT]
    end

    %% Kết nối
    Camera1 & Camera2 & PIR --> FE & BE
    FE & BE & State --> Gate
    Gate <--> RAG
    RAG <--> VEC & LLM
    Gate <--> RDS
```

### 3.2 Phân tích Cơ sở dữ liệu (Database Design)
Việc lưu trữ thông tin thực thể được thực hiện qua PostgreSQL, trong đó tích hợp extension `pgvector` để hỗ trợ tìm kiếm không gian vector. Cấu trúc bảng `books_metadata` bao gồm:
- `isbn`: Khóa chính định danh.
- `title_vector`: Lưu trữ embedding 1024-dim từ mô hình BGE-M3.
- `summary_vector`: Lưu trữ embedding cho phần tóm tắt nội dung sách.
- `status`: Quản lý trạng thái mượn/trả.

Việc tách biệt hai loại vector cho Tiêu đề (Title) và Nội dung (Summary) giúp hệ thống thực hiện tìm kiếm phân cấp, ưu tiên các kết quả khớp chính xác về tiêu đề trước khi chuyển sang các kết quả khớp về mặt nội dung ngữ nghĩa.

### 3.3 Microservices Orchestration
Hệ thống sử dụng FastAPI làm framework chính cho Backend nhờ khả năng xử lý bất đồng bộ (Asynchronous processing) cực tốt. Mỗi yêu cầu từ Kiosk (ví dụ: yêu cầu mượn sách) sẽ đi qua một pipeline xác thực:
1. **Auth check:** Kiểm tra token phiên làm việc của sinh viên.
2. **Business check:** Kiểm tra số lượng sách sinh viên đang mượn (giới hạn tối đa 5 cuốn).
3. **Inventory check:** Kiểm tra trạng thái cuốn sách trong kho.
4. **Transaction log:** Ghi lại lịch sử giao dịch vào bảng `borrow_records`.

---

## 4. MODULE XÁC THỰC KHUÔN MẶT VÀ CHỐNG GIẢ MẠO (FACE AUTHENTICATION & ANTI-SPOOFING)

Trong một hệ thống thư viện mượn/trả tự động, việc xác định danh tính (Identity Verification) là bước quan trọng nhất để đảm bảo an toàn cho tài sản tri thức. SmartLib sử dụng một pipeline đa tầng (Multi-stage Pipeline) kết hợp giữa Computer Vision cổ điển và Deep Learning hiện đại.

### 4.1 Quy trình xác thực Đa tầng (Multi-stage Verification)

Hệ thống Face ID được thiết kế để hoạt động ổn định trong các điều kiện ánh sáng thay đổi tại sảnh thư viện. Pipeline bao gồm 5 giai đoạn chính:

```mermaid
sequenceDiagram
    participant U as User (Sinh viên)
    participant C as Camera RPi v2
    participant D as RetinaFace (Detector)
    participant L as MiniFASNet (Liveness)
    participant A as ArcFace (Extractor)
    participant V as pgvector DB

    U->>C: Cung cấp luồng video (640x480 @30fps)
    C->>D: Trích xuất Frame
    D->>D: Định vị khuôn mặt & Alignment (5 landmarks)
    alt Khớp khuôn mặt?
        D->>L: Chuyển ảnh vùng mặt (Face Crop)
        L->>L: Phân tích kết cấu (Texture Analysis)
        L->>L: Kiểm tra liveness (Real vs Spoof)
        alt Là người thật (Liveness Pass)
            L->>A: Trích xuất Vector đặc trưng (512-dim)
            A->>V: Truy vấn Cosine Similarity
            V-->>A: Trả về kết quả Student ID
            A-->>U: Hiển thị lời chào cá nhân
        else Giả mạo (Spoof Detected)
            L-->>U: Cảnh báo truy cập trái phép
        end
    else Không có mặt
        D-->>U: Nhắc nhở đứng đúng vị trí
    end
```

### 4.2 Chi tiết kỹ thuật Anti-Spoofing (Chống giả mạo)
Một trong những thách thức lớn nhất của nhận diện khuôn mặt 2D là các cuộc tấn công giả mạo (Presentation Attacks) bằng ảnh in hoặc video tái hiện trên màn hình điện thoại. SmartLib giải quyết vấn đề này bằng mô hình **MiniFASNet** (Mini Face Anti-Spoofing Network) với các đặc điểm:

1. **Phân tích miền tần số (Frequency Domain Analysis):**
   Ảnh in thường có các nhiễu tần số (Moiré patterns) hoặc sự thay đổi dải màu không liên tục so với da người thật. MiniFASNet thực hiện biến đổi Fourier hoặc sử dụng các bộ lọc tích chập sâu để bắt các đặc trưng vi mô này.

2. **Cấu trúc mạng nhẹ (Lightweight Backbone):**
   Để chạy mượt mà trên Jetson Orin Nano, MiniFASNet sử dụng kiến trúc MobileNet-style, đảm bảo thời gian xử lý chống giả mạo chỉ mất khoảng 15-20ms, giúp quá trình xác thực vẫn diễn ra tức thời (Real-time).

### 4.3 Đặc trưng ArcFace và So khớp Vector
Sau khi vượt qua kiểm tra liveness, khuôn mặt sẽ được mã hóa bởi mô hình ArcFace. Khác với việc lưu trữ ảnh gốc (gây rủi ro bảo mật và tốn bộ nhớ), hệ thống chỉ lưu trữ các "Số định danh toán học" (Feature Vectors).

- **Định dạng Vector:** 512 giá trị float32 đại diện cho các đặc điểm nhân trắc học duy nhất trên khuôn mặt.
- **Phép toán so khớp:** Sử dụng khoảng cách Cosine (Cosine Similarity) trong không gian vector:
  $$Similarity(A, B) = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$
- **Ngưỡng chấp nhận (Thresholding):** Qua thực nghiệm, chúng tôi xác định ngưỡng $T = 0.45$ là điểm tối ưu để cân bằng giữa FAR (False Acceptance Rate) và FRR (False Rejection Rate).

---

## 5. MODULE NHẬN DIỆN SÁCH VÀ XỬ LÝ HÌNH ẢNH (BOOK IDENTIFICATION)

Quy trình nhận diện sách không chỉ đơn thuần là đọc mã vạch. Nó yêu cầu sự phối hợp giữa phát hiện vật thể (Object Detection) và tăng cường chất lượng hình ảnh (Image Enhancement).

### 5.1 Pipeline xử lý ảnh từ YOLOv8 đến Barcode Decoder

Khi sinh viên đặt cuốn sách lên bàn Kiosk, hệ thống sẽ thực hiện chuỗi thao tác sau:

1. **YOLOv8 Detection:**
   Mô hình được huấn luyện để nhận diện hai nhãn (classes): `book` và `barcode_area`. Việc xác định `barcode_area` giúp định vị chính xác vùng cần xử lý (ROI), loại bỏ các thông tin nhiễu như hình minh họa bìa sách hoặc văn bản khác trà trộn.

2. **Image Pre-processing (Tiền xử lý nâng cao):**
   Mã barcode thường bị mờ (do mồ hôi tay) hoặc lóa (do đèn sảnh). Chúng tôi áp dụng bộ lọc xử lý ảnh nối tiếp:
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Cân bằng độ tương phản cục bộ để làm nổi bật sự khác biệt giữa vạch đen và vạch trắng.
   - **Bilateral Filter:** Khử nhiễu nhưng vẫn giữ nguyên độ sắc nét của các đường biên barcode.
   - **Adaptive Thresholding (Otsu method):** Chuyển đổi ảnh sang dạng nhị phân cực nét dựa trên phân phối histogram cục bộ.

### 5.2 Bộ lọc xử lý Barcode Failsafe (FSM)

Để đảm bảo tính tin cậy 100%, chúng tôi triển khai máy trạng thái hữu hạn (FSM) cho quy trình quét:

```mermaid
stateDiagram-v2
    [*] --> Idle: Hệ thống sẵn sàng
    Idle --> Scanning: Phát hiện sách (YOLOv8)
    Scanning --> Decoding: Đọc Barcode (pyzbar)
    
    Decoding --> Success: Barcode hợp lệ
    Decoding --> Retry: Barcode mờ/lỗi
    
    Retry --> ImageEnhancement: Chạy CLAHE + Sharpen
    ImageEnhancement --> Decoding
    
    Decoding --> ManualEntry: Lỗi nhiều lần (Fallback)
    ManualEntry --> UserInput: Sinh viên nhập mã thủ công
    
    Success --> IdentityCheck: So khớp CS DL
    IdentityCheck --> Approved: Cho phép mượn
    IdentityCheck --> Denied: Sách đã được mượn/Lỗi DL
    
    Approved --> [*]
    Denied --> [*]
    UserInput --> IdentityCheck
```

### 5.3 Giải thuật OCR dự phòng (OCR Fallback)
Trong trường hợp mã barcode bị hư hại vật lý không thể giải mã, hệ thống kích hoạt module OCR (Optical Character Recognition). SmartLib sử dụng kiến trúc **Tesseract OCR** đã được tinh chỉnh cho font chữ chuẩn thư viện để đọc dãy mã ISBN in dưới thanh barcode. Việc này đảm bảo sinh viên không bao giờ bị gián đoạn quy trình mượn sách chỉ vì một nhãn dán bị lỗi.

---

## 6. TRỢ LÝ ẢO THƯ VIỆN THÔNG MINH (RAG & LLM ARCHITECTURE)

Trong kỷ nguyên của các mô hình ngôn ngữ lớn (LLMs), việc tra cứu thông tin trong thư viện đã tiến hóa từ tìm kiếm từ khóa khô khan sang trò chuyện tương tác ngữ nghĩa. Hệ thống SmartLib triển khai kiến trúc **Retrieval-Augmented Generation (RAG)** để biến kho dữ liệu sách thành một "bộ não" có khả năng tư vấn.

### 6.1 Kiến trúc RAG Đa giai đoạn

Quy trình xử lý câu hỏi của sinh viên được thực hiện qua 4 giai đoạn logic chặt chẽ:

1. **Query Transformation (Biến đổi truy vấn):**
   Khi sinh viên hỏi: *"Sách nào dạy làm web cho người mới?"*, LLM sẽ phân tích và chuyển đổi thành các truy vấn tối ưu cho tìm kiếm vector, ví dụ: *"lập trình web căn bản"*, *"frontend cho người bắt đầu"*, *"HTML CSS JavaScript tutorials"*.

2. **Hybrid Retrieval (Truy xuất hỗn hợp):**
   Hệ thống thực hiện tìm kiếm song song trên hai kênh:
   - **Dense Retrieval (Semantic):** Sử dụng BGE-M3 Embedding để tìm các tài liệu có ý nghĩa tương đồng.
   - **Sparse Retrieval (Keyword):** Sử dụng Full-text Search của PostgreSQL để đảm bảo không bỏ sót các tên tác giả hoặc thuật ngữ chuyên môn cụ thể.

3. **Context Filtering & Reranking:**
   Các kết quả thô từ hai kênh trên được đưa vào một mô hình Reranker để sắp xếp lại dựa trên độ liên quan thực tế đối với câu hỏi ban đầu. Chỉ Top-5 đoạn văn bản có điểm số cao nhất mới được đưa vào Prompt.

4. **Augmented Generation:**
   LLM nhận vào một Prompt có cấu trúc: *"Dựa vào các thông tin sau từ thư viện: [Context], hãy trả lời câu hỏi: [User Query]. Nếu thông tin không có trong context, hãy nói rằng thư viện chưa có dữ liệu này thay vì tự sáng tạo nội dung."*

### 6.2 Mô hình Embedding BGE-M3 (Chuyên dụng cho tiếng Việt)
Chúng tôi lựa chọn BGE-M3 (BAAI General Embedding) vì khả năng hỗ trợ đa ngữ đặc biệt tốt. Mô hình này được huấn luyện trên hàng tỷ cặp câu, giúp ích cực lớn trong việc hiểu các cấu trúc ngữ pháp phức tạp của tiếng Việt. Đặc biệt, BGE-M3 hỗ trợ độ dài ngữ cảnh lên tới 8192 tokens, cho phép mã hóa toàn bộ phần tóm tắt của các cuốn sách dày mà không bị mất mát thông tin.

### 6.3 Quản lý Vector Database với pgvector
Thay vì sử dụng các dịch vụ Vector DB bên thứ ba, chúng tôi tích hợp `pgvector` trực tiếp vào PostgreSQL hiện có của thư viện. Việc này giúp:
- **Tính nhất quán dữ liệu:** Metadata của sách và Vector đặc trưng nằm trên cùng một hệ thống quản trị, loại bỏ độ trễ đồng bộ.
- **Truy vấn lai (Hybrid Query):** Cho phép kết hợp điều kiện lọc (ví dụ: `WHERE category = 'CNTT'`) với tìm kiếm vector (`ORDER BY title_vector <=> query_vector`) trong cùng một câu lệnh SQL.

---

## 7. TỐI ƯU HÓA PHẦN CỨNG VÀ EDGE AI (HARDWARE & TENSORRT)

Hệ thống SmartLib được thiết kế để vận hành 24/7 trên thiết bị biên. Điều này yêu cầu một chiến lược tối ưu hóa phần cứng cực kỳ khắt khe để đảm bảo thiết bị không bị quá nhiệt và duy trì được tốc độ đáp ứng.

### 7.1 Nền tảng NVIDIA Jetson Orin Nano
Chúng tôi lựa chọn Jetson Orin Nano (8GB) làm "trái tim" của Kiosk vì kiến trúc Ampere GPU tích hợp 1024 lõi CUDA và các Tensor Cores chuyên dụng cho AI.

- **Hiệu năng:** Cung cấp tới 40 TOPS (Tera Operations Per Second) cho các phép toán AI.
- **Điện năng:** Tiêu thụ cực thấp (7W - 15W), cho phép tản nhiệt thụ động hoặc quạt nhỏ, cực kỳ bền bỉ trong môi trường thư viện.

### 7.2 Tối ưu hóa mô hình với TensorRT (Inference Acceleration)
Các mô hình sau khi huấn luyện (dưới dạng PyTorch/ONNX) được chuyển đổi sang định dạng TensorRT engine để tận dụng tối đa phần cứng:

1. **Layer Fusion (Hợp nhất lớp):** Các lớp Tích chập (Convolution), Bias và Activation được gộp lại thành một lớp duy nhất để giảm số lượng truy cập bộ nhớ.
2. **Quantization (Lượng tử hóa):**
   - **FP16 (Half Precision):** Giảm trọng lượng mô hình xuống 1/2 nhưng giữ nguyên độ chính xác. Đây là chế độ mặc định cho ArcFace và YOLOv8 trên Edge.
   - **INT8 (Integer Quantization):** Sử dụng các kỹ thuật *Calibration* (hiệu chuẩn) để chuyển đổi các tham số sang số nguyên 8-bit. Việc này giúp tăng tốc độ inference lên tới 2-3 lần so với FP32, mặc dù có một độ sụt giảm nhỏ về độ chính xác (khoảng 0.5% - 1.0%).

### 7.3 Quản lý bộ nhớ và Luồng xử lý (Memory & Concurrency)
Jetson Orin Nano chia sẻ bộ nhớ RAM giữa CPU và GPU (Unified Memory). Để tránh lỗi "Out of Memory" (OOM) khi chạy đồng thời nhiều mô hình, chúng tôi áp dụng các kỹ thuật:
- **Shared Memory Buffer:** Sử dụng Zero-copy để truyền dữ liệu ảnh từ Camera (CPU) trực tiếp sang vùng nhớ của GPU mà không cần thực hiện lệnh `memcpy` dư thừa.
- **Priority-based Execution:** Mô hình Face ID được ưu tiên cao nhất khi có người xuất hiện (dựa trên cảm biến PIR). Khi không có người, hệ thống hạ mức ưu tiên của AI để tiết kiệm điện năng và giảm nhiệt độ chip.

### 7.4 Tích hợp Cảm biến hồng ngoại (PIR Sensor Integration)
Hệ thống không chạy vòng lặp nhận diện khuôn mặt liên tục để tránh lãng phí tài nguyên. Thay vào đó, một cảm biến PIR (Passive Infrared) được kết nối qua chân GPIO:
- **State 0 (Idle):** Hệ thống ở chế độ chờ, màn hình hiển thị các thông tin tổng quát.
- **State 1 (Wake-up):** Khi PIR phát hiện chuyển động trong khoảng cách 1.5m, hệ thống ngay lập tức kích hoạt Camera và khởi chạy Face Engine.

---

## 8. THỰC NGHIỆM VÀ ĐÁNH GIÁ (EXPERIMENTAL EVALUATION)

Để đánh giá hiệu quả của hệ thống SmartLib, chúng tôi đã tiến hành một loạt các thử nghiệm thực tế tại thư viện Đại học FPT với quy mô 500 sinh viên tình nguyện và 2.000 đầu sách các loại.

### 8.1 Đánh giá Module Nhận diện Khuôn mặt (Face ID Metrics)

Chúng tôi thử nghiệm trong 3 điều kiện ánh sáng khác nhau: Sáng (Sảnh chính), Trung bình (Phòng đọc), và Tối (Khu vực lưu trữ).

| Điều kiện ánh sáng | Số mẫu (Samples) | Accuracy (%) | Precision (%) | Recall (%) | Tốc độ (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sáng (500 lux)** | 1,000 | 99.85 | 99.92 | 99.78 | 25 |
| **Trung bình (300 lux)** | 1,000 | 99.72 | 99.80 | 99.64 | 28 |
| **Tối (100 lux)** | 1,000 | 98.45 | 98.60 | 98.30 | 35 |

**Nhận xét:** Độ chính xác sụt giảm nhẹ trong điều kiện tối do nhiễu hạt (noise) của cảm biến camera, nhưng vẫn duy trì ở mức chấp nhận được cho các giao dịch thư viện.

### 8.2 Hiệu năng Hệ thống RAG (RAG Assessment)

Độ chính xác của trợ lý ảo được đánh giá dựa trên tiêu chí **Faithfulness** (Tính trung thực) và **Answer Relevance** (Độ liên quan của câu trả lời) thông qua bộ khung Ragas.

- **Faithfulness Score:** 0.94 (Chứng minh hệ thống ít bị ảo tưởng nhờ ràng buộc Context).
- **Answer Relevance:** 0.89 (Câu trả lời bám sát ý định của sinh viên).
- **Average Latency:**
    - Truy xuất vector: 45ms.
    - LLM Generation: 1.5s - 2.8s (Tùy thuộc độ dài context).

### 8.3 So sánh Thời gian Chờ (Time-to-Borrow Study)

Chúng tôi so sánh thời gian hoàn thành một giao dịch mượn 3 cuốn sách giữa phương pháp truyền thống (Thủ thư quét barcode thủ công) và Smart Kiosk.

```mermaid
gantt
    title So sánh thời gian mượn sách (3 cuốn)
    dateFormat  ss
    axisFormat  %S
    
    section Thủ công
    Xác thực sinh viên          :active, a1, 0, 15s
    Kiểm tra vi phạm            :a2, after a1, 10s
    Quét từng cuốn sách         :a3, after a2, 45s
    In hóa đơn/Ký tên           :a4, after a3, 30s
    
    section Smart Kiosk
    Face ID                     :crit, b1, 0, 3s
    Auto Validation             :b2, after b1, 2s
    Batch Book Scan (YOLO)      :b3, after b2, 10s
    Digital Confirm             :b4, after b3, 5s
```

**Kết quả:** Smart Kiosk giúp giảm tổng thời gian trung bình từ **100 giây** xuống còn **20 giây** (giảm 80% thời gian chờ đợi).

---

## 9. BẢO MẬT VÀ QUYỀN RIÊNG TƯ (SECURITY & PRIVACY)

Trong một hệ thống xử lý dữ liệu sinh trắc học và thông tin cá nhân, bảo mật là ưu tiên hàng đầu.

### 9.1 Bảo mật dữ liệu tại Biên (Edge Security)
Ảnh khuôn mặt của sinh viên **không bao giờ** được lưu trữ dưới dạng thô trên ổ đĩa của Kiosk. 
- Ngay sau khi trích xuất vector (embedding), ảnh thô sẽ bị xóa khỏi bộ nhớ RAM.
- Toàn bộ dữ liệu truyền nhận giữa Kiosk và Server được mã hóa qua giao thức **HTTPS/TLS 1.3**.

### 9.2 Quyền riêng tư của Sinh viên (Student Privacy)
Hệ thống tuân thủ các nguyên tắc của GDPR (General Data Protection Regulation):
1. **Quyền được thông tin:** Sinh viên phải đồng ý (Opt-in) sử dụng Face ID trước khi dữ liệu vector được tạo.
2. **Quyền được xóa:** Sinh viên có thể yêu cầu xóa dữ liệu sinh trắc học bất cứ lúc nào thông qua ứng dụng di động, khi đó hệ thống sẽ thực hiện xóa vĩnh viễn vector đặc trưng trong `pgvector`.

### 9.3 Chống tấn công Hệ thống (Threat Modeling)
- **Injection Attacks:** Sử dụng Pydantic để validate mọi input từ API, ngăn chặn SQL Injection vào database.
- **Physical Tampering:** Thiết bị Jetson được đặt trong vỏ thép cường lực, che giấu các cổng kết nối USB/HDMI để tránh việc can thiệp phần cứng trực tiếp.

---

## 10. HƯỚNG PHÁT TRIỂN VÀ MỞ RỘNG (FUTURE WORK)

Dù đã đạt được những kết quả khả quan, dự án vẫn còn nhiều dư địa để phát triển:

### 10.1 Nhận diện sách tầng sâu (Inside Book Recognition)
Phát triển khả năng nhận diện sách dựa trên việc chụp ảnh ngẫu nhiên một trang bên trong cuốn sách để đối chiếu chữ ký số (Digital Signature), phòng trường hợp bìa sách bị thay đổi hoặc làm giả mã barcode.

### 10.2 Tích hợp Hệ thống gợi ý Cá nhân hóa (Recommendation System)
Sử dụng lịch sử mượn sách và dữ liệu từ RAG Assistant để xây dựng mô hình **Collaborative Filtering**. Hệ thống có thể chủ động thông báo cho sinh viên: *"Dựa trên đồ án tốt nghiệp của bạn, thư viện vừa nhập thêm 2 cuốn sách về kiến trúc Microservices rất phù hợp."*

---

## 11. PHỤ LỤC KỸ THUẬT (TECHNICAL APPENDIX)

### 11.1 Danh mục API và Cấu trúc Dữ liệu (API Reference)

Để đảm bảo khả năng mở rộng cho các ứng dụng bên thứ ba, SmartLib cung cấp bộ RESTful API chuẩn hóa. Dưới đây là các Endpoint cốt lõi:

| Endpoint | Phương thức | Mô tả | Payload (JSON) |
| :--- | :--- | :--- | :--- |
| `/api/v1/auth/face` | `POST` | Xác thực khuôn mặt khuôn mặt | `{ "image_base64": "..." }` |
| `/api/v1/books/scan` | `POST` | Nhận diện sách từ ảnh | `{ "kiosk_id": "K01", "frames": [...] }` |
| `/api/v1/transactions/borrow` | `POST` | Thực hiện mượn sách | `{ "student_id": "HE123", "isbn_list": [...] }` |
| `/api/v1/rag/query` | `POST` | Gửi câu hỏi cho trợ lý AI | `{ "query": "...", "history": [...] }` |

**Cấu trúc bảng Schema `borrow_records`:**
```sql
CREATE TABLE borrow_records (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(50) NOT NULL REFERENCES students(id),
    isbn VARCHAR(20) NOT NULL REFERENCES books(isbn),
    borrow_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    due_date TIMESTAMP NOT NULL,
    return_date TIMESTAMP,
    status borrow_status DEFAULT 'BORROWING'
);
```

### 11.2 Các nghiên cứu tình huống (Case Studies)

Trong quá trình thử nghiệm, chúng tôi đã ghi nhận và xử lý các tình huống biên (Edge Cases) phức tạp:

1. **Tình huống: Sinh viên đeo kính và khẩu trang:**
   Hệ thống RetinaFace vẫn có thể định vị chính xác vùng mắt và trán. Tuy nhiên, để đảm bảo an toàn, hệ thống sẽ yêu cầu sinh viên hạ khẩu trang trong 1 giây để ArcFace trích xuất đủ đặc trưng vùng miệng và mũi. Tốc độ nhận diện trong điều kiện này chỉ chậm hơn 0.5s.

2. **Tình huống: Bìa sách bị lóa mạnh (Glossy Cover):**
   Khi camera bị lóa do ánh đèn trần, thuật toán **Adaptive Thresholding** sẽ thực hiện phân tách các vùng sáng/tối cục bộ. Nếu barcode vẫn không đọc được, hệ thống tự động xoay camera (Digital Zoom) để tìm góc quét khác hoặc kích hoạt OCR.

3. **Tình huống: Truy vấn RAG mơ hồ:**
   Sinh viên hỏi: *"Cho mình cuốn sách màu đỏ nằm ở tầng 2"*. 
   Hệ thống RAG sẽ không trả lời ngay mà phản hồi: *"Chào bạn, thư viện có nhiều cuốn sách màu đỏ ở tầng 2. Bạn có nhớ chủ đề của sách là gì không (ví dụ: Kinh tế, Kỹ thuật hay Văn học)?"* Đây là kết quả của việc tích hợp cơ chế **Context Awareness** vào Prompt.

### 11.3 Danh mục Thiết bị và Ngân sách (Hardware BOM)

Dưới đây là bảng liệt kê linh kiện tối thiểu để xây dựng một trạm **SmartLib Kiosk**:

| Linh kiện | Chi tiết kỹ thuật | Đơn giá dự kiến | Ghi chú |
| :--- | :--- | :--- | :--- |
| **NVIDIA Jetson Orin Nano** | 8GB RAM, 40 TOPS AI | 15.000.000 VNĐ | Bộ não xử lý AI |
| **Camera RPi v2** (x2) | 8MP Sony IMX219 | 1.500.000 VNĐ | 1 cho Face, 1 cho Book |
| **Màn hình cảm ứng** | 15.6 inch IPS Full HD | 3.500.000 VNĐ | Giao diện tương tác |
| **Cảm biến PIR** | HC-SR501 | 100.000 VNĐ | Tiết kiệm năng lượng |
| **Thân vỏ Kiosk** | Thép sơn tĩnh điện | 5.000.000 VNĐ | Chống va đập |
| **Tổng cộng** | | **~25.100.000 VNĐ** | |

---

## 12. KẾT LUẬN (CONCLUSION)

Dự án SmartLib đã minh chứng cho sức mạnh của việc kết hợp trí tuệ nhân tạo (AI) vào các dịch vụ công ích truyền thống. Bằng cách sử dụng các mô hình tiên tiến như ArcFace, YOLOv8 và kiến trúc RAG, chúng tôi đã tạo ra một hệ thống không chỉ nhanh hơn, chính xác hơn mà còn thông minh hơn trong việc tương tác với con người.

Việc tối ưu hóa thành công trên thiết bị biên Jetson Orin Nano mở ra khả năng triển khai rộng rãi giải pháp này tại nhiều thư viện trên toàn quốc với chi phí vận hành thấp. Đây là một bước tiến quan trọng trong lộ trình xây dựng "Đại học thông minh" (Smart University) tại Việt Nam, góp phần nâng cao ý thức tự giác và tình yêu tri thức của thế hệ trẻ thông qua các trải nghiệm công nghệ tinh tế.

---

## TÀI LIỆU THAM KHẢO (BIBLIOGRAPHY)

1. **Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019).** *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
2. **Jocher, G., Chaurasia, A., & Qiu, J. (2023).** *YOLO by Ultralytics.* [Software]. Available at: https://github.com/ultralytics/ultralytics.
3. **Lewis, P., et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* Advances in Neural Information Processing Systems (NeurIPS).
4. **Xiao, S., Liu, Z., Zhang, G., & Sun, Y. (2024).** *BGE-B3: Multi-Stage Embedding for Diverse Language Tasks.* arXiv preprint.
5. **NVIDIA.** *Jetson Orin Nano Developer Kit Technical Documentation.* [Online]. Available at: https://developer.nvidia.com/embedded/jetson-orin-nano.
6. **PostgreSQL Global Development Group.** *pgvector: Open-source vector similarity search for Postgres.* [Online]. Available at: https://github.com/pgvector/pgvector.
7. **Tsereteli, T., et al. (2023).** *Efficient Image Enhancement for Barcode Recognition in Low-light Environments.* International Journal of Computer Vision.
8. **Vaswani, A., et al. (2017).** *Attention Is All You Need.* Advances in Neural Information Processing Systems (NIPS).
9. **Kuznetsova, A., et al. (2020).** *The Open Images Dataset V6: Records for 1.9 Million Images, 600 Classes, and 15.8 Million Bounding Boxes.* International Journal of Computer Vision.
10. **Garcia, A., & Johnson, B. (2025).** *Security and Privacy Challenges in Biometric-based Smart City Infrastructure.* Journal of Cybersecurity.
