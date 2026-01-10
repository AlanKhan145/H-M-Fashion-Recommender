# H&M Fashion Recommender

Dự án này xây dựng hệ thống gợi ý sản phẩm cho bài toán **H&M Personalized Fashion Recommendations** (Kaggle). README này tập trung **hướng dẫn cài đặt và chạy dự án** ở môi trường local hoặc Kaggle Notebook.

---

## 1. Yêu cầu hệ thống

* **Python**: 3.8 – 3.11 (khuyến nghị 3.9)
* **Hệ điều hành**: Windows / Linux / macOS
* **RAM**: tối thiểu 8GB (16GB+ khuyến nghị)
* **GPU** (tuỳ chọn): NVIDIA GPU (nếu dùng PyTorch / CNN / ViT)

---

## 2. Tạo môi trường ảo (khuyến nghị)

### Cách 1: dùng `venv`

```bash
python -m venv venv
```

Kích hoạt môi trường:

* **Windows**:

```bash
venv\Scripts\activate
```

* **Linux / macOS**:

```bash
source venv/bin/activate
```

---

## 3. Cài đặt thư viện cần thiết

### 3.1. Thư viện cơ bản

```bash
pip install -U pip
pip install numpy pandas scipy scikit-learn tqdm pillow matplotlib
```

### 3.2. Machine Learning / Recommendation

```bash
pip install implicit lightfm
```

### 3.3. Xử lý ảnh & Deep Learning (tuỳ chọn)

Nếu sử dụng **ResNet / ViT / DINO**:

```bash
pip install torch torchvision torchaudio
pip install timm
```

> ⚠️ Lưu ý: với GPU, hãy cài PyTorch đúng CUDA version theo hướng dẫn tại [https://pytorch.org](https://pytorch.org)

---

## 4. Cấu trúc thư mục (đề xuất)

```text
h-m-fashion-recommender/
│
├── data/
│   ├── articles.csv
│   ├── customers.csv
│   ├── transactions_train.csv
│   └── images/
│
├── notebooks/
│   └── h-m-fashion-recommender-01.ipynb
│
├── outputs/
│   ├── features/
│   └── submissions/
│
├── requirements.txt
└── README.md
```

---

## 5. Chuẩn bị dữ liệu

1. Tải dữ liệu từ Kaggle:
   **H&M Personalized Fashion Recommendations**

2. Giải nén và đặt vào thư mục `data/`

3. Đảm bảo các file sau tồn tại:

* `articles.csv`
* `customers.csv`
* `transactions_train.csv`
* `images/` (nếu dùng image-based recommender)

---

## 6. Chạy Notebook

### 6.1. Mở Jupyter Notebook

```bash
jupyter notebook
```

Hoặc:

```bash
jupyter lab
```

Mở file:

```text
notebooks/h-m-fashion-recommender-01.ipynb
```

### 6.2. Thứ tự chạy

1. Load & preprocess dữ liệu
2. Trích xuất feature (text / image / embedding)
3. Huấn luyện mô hình (Content-based / KNN / ALS / Hybrid)
4. Sinh **Top-12 recommendations**
5. Xuất file submission `.csv` hoặc `.pkl`

---

## 7. Tạo file submission

Định dạng chuẩn Kaggle:

```csv
customer_id,prediction
00000dba,0706016001 0706016002 ...
```

* Mỗi khách hàng: **tối đa 12 article_id**
* Các article_id cách nhau bằng **dấu cách**

---

## 8. Đánh giá mô hình (offline)

Các metric thường dùng:

* `MAP@12` (metric chính của Kaggle)
* `Precision@12`
* `Recall@12`
* `F1@12`

> Lưu ý: Kaggle **không dùng Accuracy truyền thống** cho bài toán recommender.

---

## 9. Chạy trên Kaggle Notebook (khuyến nghị)

1. Upload notebook `.ipynb`
2. Add dataset **H&M Personalized Fashion Recommendations**
3. Bật GPU (nếu cần)
4. Run All

> Kaggle giúp tránh lỗi RAM và tăng tốc I/O đáng kể.

---



---

📌 *README này tập trung vào cài đặt & chạy dự án. Có thể mở rộng thêm phần mô tả thuật toán nếu cần.*
