# 🖥️ DSP Framework GUI Specification

Tài liệu này đặc tả thiết kế và kiến trúc cho Giao diện người dùng (GUI) của hệ thống mô phỏng DSP. GUI đóng vai trò là một "Bảng điều khiển ảo" (Virtual Dashboard), kết nối trực tiếp với backend Python (các class `DSPPipeline` và `DSPBlock`).

## 1. Lựa chọn Công nghệ (Tech Stack)

Để tối ưu giữa hiệu năng render đồ thị và tốc độ phát triển bằng Python, dưới đây là các lựa chọn được đề xuất:

* **Framework cốt lõi (Desktop GUI):** `PyQt6` hoặc `PySide6`. 
  *(Nếu muốn làm Web-based Dashboard nhanh: dùng `Streamlit` hoặc `Dash` by Plotly).*
* **Giao diện Node-based (Kéo thả khối):** Thư viện `NodeGraphQt` (tương thích PyQt) để tạo giao diện nối dây giữa các block.
* **Đồ thị (Real-time Plotting):** `pyqtgraph` (Rất nhẹ và nhanh cho vẽ array NumPy theo thời gian thực, vượt trội hơn Matplotlib khi render realtime).

## 2. Bố cục màn hình (Screen Layout)

Giao diện được chia thành 4 vùng chính (Panels):

### 2.1. Panel Trái: Thư viện Block (Block Library)
* **Mô tả:** Chứa danh sách các `DSPBlock` có sẵn trong framework, được chia theo nhóm.
* **Các nhóm:**
  * *Sources:* PRBS Generator, Sine/Cosine, Custom Waveform.
  * *Channels:* AWGN, Multipath Fading, Frequency Offset, S-Parameter (`.s4p`).
  * *Filters & Comms:* Root Raised Cosine (RRC), QAM Mapper, Equalizers.
  * *Loops (Super-Blocks):* PLL, CDR.
* **Tương tác:** Người dùng có thể kéo (drag) một block từ thư viện thả vào Canvas.

### 2.2. Panel Giữa: Bảng thiết kế (Pipeline Canvas)
* **Mô tả:** Một giao diện dạng Node/Graph hoặc Dạng danh sách tuyến tính (Linear List) đại diện cho `DSPPipeline`.
* **Tính năng:**
  * Cho phép thêm/xóa/đổi chỗ các khối.
  * Hiển thị luồng dữ liệu $Tx \rightarrow Channel \rightarrow Rx$.
  * Các node có các điểm "Probe" (Điểm đo) để trích xuất tín hiệu đưa ra đồ thị.

### 2.3. Panel Phải: Bảng Thuộc tính (Property Inspector)
* **Mô tả:** Khi bấm vào một Block trên Canvas, panel này hiện ra các tham số (parameters) của block đó.
* **Tương tác Backend:** Khi người dùng thay đổi giá trị và ấn "Apply" (hoặc thay đổi qua Slider), GUI sẽ gọi hàm `block.update_params(**new_kwargs)`.
* **Ví dụ:**
  * Chọn khối *AWGN*: Hiển thị Slider cho `snr_db` (0 đến 40 dB).
  * Chọn khối *SParameterChannel*: Nút "Browse..." để chọn file `.s4p`.

### 2.4. Panel Dưới / Cửa sổ nổi: Khu vực Đồ thị (Scopes & Visualization)
* **Mô tả:** Chứa các màn hình hiển thị tín hiệu tại các điểm "Probe" được chọn.
* **Các loại Đồ thị:**
  * **Time-Domain Oscilloscope:** Xem biên độ I/Q theo thời gian.
  * **Spectrum Analyzer (FFT):** Xem phổ tần số, tự động tính PSD (Power Spectral Density).
  * **Constellation Diagram:** Đồ thị phân tán (Scatter plot) vẽ trục I và Q.
  * **Eye Diagram:** Đồ thị mắt (cực kỳ quan trọng để phân tích S-Parameter và CDR).

## 3. Luồng hoạt động (Execution Flow)

Hệ thống GUI sẽ không làm chết (freeze) giao diện khi chạy mô phỏng. Nó cần một luồng (Thread) riêng cho DSP.

1. **Initialization:** Người dùng kéo thả và set tham số $\rightarrow$ GUI khởi tạo list các `DSPBlock` và bọc trong `DSPPipeline`.
2. **Simulation Mode:**
   * **Single Shot:** Bấm "Run", pipeline xử lý một array tín hiệu lớn (ví dụ $10^6$ samples), sau đó update đồ thị.
   * **Continuous (Streaming) Mode:** Một `QThread` chạy ngầm, liên tục tạo các chunk array nhỏ (ví dụ $4096$ samples), đẩy qua `pipeline.process(chunk)`, và gửi kết quả lên GUI để update đồ thị `pyqtgraph` ở mức 30 FPS.
3. **Parameter Tuning:** Trong lúc Streaming, nếu người dùng kéo thanh trượt `SNR`, GUI gọi `awgn.update_params(snr_db=new_val)`. Do backend xử lý chunk-by-chunk, đồ thị Constellation sẽ lập tức mờ đi hoặc nét lên (realtime).

## 4. Đặc tả API kết nối GUI và Core

Để GUI hoạt động, chúng ta cần bổ sung một class quản lý (Controller) làm trung gian:

```python
class SimulationController:
    def __init__(self):
        self.pipeline = DSPPipeline()
        self.probes = {} # Lưu trữ tín hiệu đầu ra tại các bước

    def build_pipeline_from_gui_config(self, config_list):
        # Chuyển đổi JSON/Dict từ Canvas thành đối tượng Python
        pass

    def run_step(self, input_chunk):
        # Chạy pipeline và lưu lại tín hiệu để vẽ
        current_data = input_chunk
        for idx, block in enumerate(self.pipeline.blocks):
            current_data = block.process(current_data)
            self.probes[f"block_{idx}"] = current_data
        return current_data
```

## 5. Các bước triển khai (Implementation Roadmap)

* **Bước 1:** Xây dựng khung GUI cơ bản (PyQt6) với Property Inspector bên phải và 1 biểu đồ Plot mẫu. Dùng Hard-code pipeline để test.
* **Bước 2:** Tích hợp `pyqtgraph` để vẽ Constellation và Time-domain.
* **Bước 3:** Thêm tính năng "Continuous Mode" dùng `QThread` để mô phỏng tính thời gian thực.
* **Bước 4:** Áp dụng `NodeGraphQt` để làm tính năng kéo thả linh hoạt.