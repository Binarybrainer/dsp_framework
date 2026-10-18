# ⚡ Hybrid Performance Strategy: Python & C++ Integration

Trong phát triển DSP, có một nghịch lý: **Python rất chậm với các vòng lặp, nhưng lại cực nhanh khi thao tác mảng (Matrix/Array)**. Để xây dựng một Framework vừa dễ phát triển vừa có tốc độ thực thi thời gian thực, chúng ta áp dụng **Chiến lược Kết hợp (Hybrid Strategy)**.

Tài liệu này định nghĩa cách phân loại thuật toán và tiêu chuẩn tích hợp C++ vào Python để loại bỏ hoàn toàn các nút thắt cổ chai (bottlenecks).

## 1. Phân loại Phép toán DSP (DSP Operation Classification)

Mọi khối `DSPBlock` trong framework phải được phân loại vào một trong hai nhóm sau trước khi lập trình:

### Nhóm A: Các khối Vector hóa (Feed-forward / Vectorized)
* **Đặc điểm:** Không có trạng thái phụ thuộc lẫn nhau giữa các mẫu liên tiếp (Mẫu $n$ không phụ thuộc vào kết quả của mẫu $n-1$).
* **Ví dụ:** AWGN (Nhiễu trắng), Modulator/Demodulator, FFT/IFFT, Khối nhân ma trận MIMO (Zero-Forcing), Bộ lọc FIR.
* **Quyết định triển khai:** **100% Native Python (NumPy/SciPy).** 
* **Lý do:** Khi bạn gọi `np.fft` hoặc `scipy.signal.lfilter`, Python sẽ đẩy dữ liệu trực tiếp xuống thư viện lõi C/Fortran của NumPy. Viết lại các khối này bằng C++ là lãng phí thời gian và hầu như không tăng thêm hiệu năng.

### Nhóm B: Các khối Phản hồi (Feedback / Scalar Loops)
* **Đặc điểm:** Yêu cầu xử lý nối tiếp từng mẫu (sample-by-sample). Đầu ra của hàm phụ thuộc vào biến trạng thái (state) được cập nhật ở vòng lặp trước đó.
* **Ví dụ:** Phase-Locked Loop (PLL), Clock and Data Recovery (CDR), Decision Feedback Equalizer (DFE), IIR Filters.
* **Quyết định triển khai:** **C++ Core + pybind11.**
* **Lý do:** Python Interpreter có độ trễ lớn và dính Global Interpreter Lock (GIL). Một vòng lặp `for i in range(len(data)):` trong Python chậm hơn C++ từ 50x - 100x. Các khối này bắt buộc phải viết bằng C++20.

## 2. Giải pháp Cốt lõi: Zero-Copy Memory (pybind11)

Lỗi phổ biến nhất khi kết hợp Python và C++ là để bộ nhớ bị "copy" (nhân bản) khi truyền dữ liệu giữa 2 ngôn ngữ. Framework của chúng ta sử dụng cơ chế **Zero-Copy Memory** qua thư viện `pybind11`.

### Cách hoạt động:
1. Python cấp phát một mảng `numpy.ndarray` chứa tín hiệu (Ví dụ: 1 triệu mẫu Complex).
2. Khi gọi hàm `pll.process(data)` của C++ từ Python, `pybind11` sẽ không copy mảng này. Thay vào đó, nó truyền **Con trỏ bộ nhớ (Memory Pointer)** của NumPy Array thẳng xuống hàm C++.
3. Hàm C++ nhận `std::complex<float>*`, xử lý vòng lặp nội bộ siêu tốc, và ghi trực tiếp đè lên vùng nhớ đó (hoặc ghi ra một buffer đích do Python chuẩn bị sẵn).

### Đoạn code minh họa (C++ Backend cho Python):
```cpp
// Trích xuất trong src/bindings/python/bind_blocks.cpp
.def("process", [](PLLBlock& self, py::array_t<std::complex<float>> input) {
    // 1. Lấy thông tin buffer trực tiếp từ NumPy (Không Copy)
    py::buffer_info buf = input.request();
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    size_t num_samples = buf.shape[0];

    // 2. Chạy vòng lặp for nội bộ tốc độ cao của C++
    self.process_in_place(ptr, num_samples);
})
```

## 3. Quy trình Tối ưu hóa Thực tế (Migration Workflow)

Để không mất thời gian viết C++ cho những thứ không cần thiết, Framework áp dụng quy trình phát triển 3 bước (Top-Down Profiling):

* **Bước 1: Viết 100% bằng Python (Rapid Prototyping):** Viết mọi thứ, kể cả PLL và CDR bằng Python. Mục tiêu là để xác minh logic toán học, đồ thị và hệ số PID là chính xác.
* **Bước 2: Đo lường (Profiling):** Chạy công cụ `cProfile` hoặc `line_profiler` của Python trên chuỗi Pipeline. 
  ```bash
  python -m cProfile -s cumtime run_simulation.py
  ```
  Công cụ này sẽ chỉ ra chính xác Block nào đang chiếm 90% thời gian CPU.
* **Bước 3: Dịch thuật (Targeted C++ Rewrite):** Chỉ lấy những Block chậm nhất (như `PLLBlock`) được phát hiện ở Bước 2, chuyển mã thuật toán sang C++, gắn thẻ pybind11, và thay thế nóng vào chuỗi `DSPPipeline`. Phía người dùng giao diện GUI sẽ không nhận ra sự khác biệt về API, chỉ thấy tốc độ tăng vọt.

## 4. Tổng kết

Bằng cách tuân thủ nghiêm ngặt **Chiến lược Hybrid** này, Framework của chúng ta đạt được sự cân bằng hoàn hảo:
* **Thời gian phát triển nhanh:** Nhờ 80% code cấu trúc, UI và Pipeline viết bằng Python.
* **Hiệu năng Real-time:** Nhờ 20% code (các vòng lặp thuật toán khó) được xử lý trực tiếp trên thanh ghi CPU bằng C++20.