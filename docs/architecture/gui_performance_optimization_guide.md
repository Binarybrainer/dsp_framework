# 🚀 GUI Performance Optimization Guide (Chống giật lag)

Khi xây dựng GUI cho mô hình DSP theo thời gian thực (Real-time Streaming), vấn đề lớn nhất thường gặp là **GUI bị đơ (freeze)** hoặc **khung hình bị giật (stuttering)**. Nguyên nhân chủ yếu là do luồng chính của giao diện (Main GUI Thread) bị chặn bởi các phép toán ma trận nặng của NumPy hoặc quá trình render hàng triệu điểm dữ liệu.

Tài liệu này đặc tả 4 trụ cột tối ưu hóa để đảm bảo GUI chạy mượt mà ở mức 30-60 FPS ngay cả khi mô phỏng tín hiệu tốc độ cao.

---

## 1. Kiến trúc Đa luồng (Threading Architecture)

Tuyệt đối không chạy hàm `pipeline.process()` trên Main GUI Thread. Chúng ta phải tách biệt hoàn toàn **Luồng Giao diện (Frontend)** và **Luồng Xử lý (Backend)**.

### 1.1. Sử dụng `QThread` (Worker Pattern)
Sử dụng `QThread` kết hợp với `QObject` (Worker) và `Signals/Slots` của PyQt/PySide để giao tiếp an toàn giữa các luồng.

```python
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import numpy as np

class DSPWorker(QObject):
    # Signal gửi kết quả NumPy array về GUI (Truyền tham chiếu, rất nhanh)
    data_ready = pyqtSignal(np.ndarray, np.ndarray) 

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.is_running = True

    def run_simulation(self):
        """Hàm chạy vòng lặp DSP vô tận trong luồng riêng"""
        while self.is_running:
            # 1. Tạo hoặc lấy chunk dữ liệu mới
            input_chunk = np.random.randn(4096) + 1j * np.random.randn(4096)
            
            # 2. Xử lý qua pipeline (Tính toán nặng nằm ở đây)
            output_chunk = self.pipeline.process(input_chunk)
            
            # 3. Gửi tín hiệu về GUI để vẽ
            self.data_ready.emit(input_chunk, output_chunk)
            
            # (Tùy chọn) time.sleep() cực nhỏ để tránh ngốn 100% CPU nếu xử lý quá nhanh
```

## 2. Tối ưu hóa Vẽ Biểu đồ (Plotting Optimization)

Thư viện `pyqtgraph` rất nhanh, nhưng nếu ép nó vẽ 1 triệu điểm mỗi khung hình, nó vẫn sẽ giật. Cần cấu hình ép kiểu render và giảm lấy mẫu (downsampling).

### 2.1. Bật tăng tốc phần cứng (Hardware Acceleration)
Kích hoạt OpenGL cho `pyqtgraph` ở ngay đầu file script (trước khi khởi tạo QApplication).

```python
import pyqtgraph as pg
# Bật render bằng OpenGL (GPU) thay vì CPU
pg.setConfigOptions(useOpenGL=True, antialias=False)
```

### 2.2. Giảm lấy mẫu màn hình (Downsampling)
Màn hình máy tính thường chỉ có độ rộng ~1920 pixels. Nếu mảng dữ liệu dài $10^5$ mẫu, việc vẽ tất cả là vô nghĩa. `pyqtgraph` hỗ trợ gộp điểm tự động.

```python
# Cấu hình trên đối tượng PlotDataItem
self.plot_item.setDownsampling(ds=True, auto=True, method='peak')
self.plot_item.setClipToView(True) # Chỉ vẽ những điểm nằm trong khung hình hiện tại
```

### 2.3. Tắt Tự động căn chỉnh trục (Disable Auto-Ranging)
Việc GUI liên tục tính toán lại min/max của trục X/Y mỗi lần có dữ liệu mới sẽ làm tụt FPS trầm trọng.

```python
self.plot_widget.setXRange(0, 4096, padding=0)
self.plot_widget.setYRange(-2.0, 2.0, padding=0)
self.plot_widget.disableAutoRange() # Tắt tính năng tự giãn trục
```

## 3. Tối ưu hóa Dữ liệu truyền tải (Data Throttling)

Luồng DSP (Backend) có thể xử lý xong 500 block mỗi giây, nhưng mắt người và màn hình chỉ cần 30 - 60 khung hình/giây (FPS). Không được bắt GUI vẽ mọi block mà backend tính ra.

### 3.1. Cơ chế Giới hạn Frame Rate (Decimation)
Backend cứ chạy ở tốc độ tối đa, nhưng GUI chỉ lấy mẫu dữ liệu mới nhất (latest snapshot) dựa trên một `QTimer` (ví dụ: kích hoạt mỗi 30ms).

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.latest_data = None
        
        # Timer cập nhật GUI ở mức ~30 FPS (33ms)
        self.gui_timer = QTimer()
        self.gui_timer.timeout.connect(self.update_plots)
        self.gui_timer.start(33)

    def on_data_ready(self, _, out_data):
        # Slot nhận dữ liệu từ Worker. 
        # Chỉ LƯU THAM CHIẾU dữ liệu mới nhất, KHÔNG VẼ ở đây.
        self.latest_data = out_data

    def update_plots(self):
        # GUI tự động vẽ dữ liệu mới nhất mỗi 33ms
        if self.latest_data is not None:
            self.plot_curve.setData(self.latest_data.real)
            self.latest_data = None # Chờ frame tiếp theo
```

## 4. Quản lý Bộ nhớ (Memory & Garbage Collection)

Python Garbage Collector (GC) là sát thủ của Real-time. Nếu bạn liên tục tạo array NumPy mới (`np.zeros(...)`) trong vòng lặp `process()`, rác bộ nhớ sẽ tích tụ. Khi GC dọn rác, GUI sẽ bị "khựng" (hiccup) mất vài chục ms.

### 4.1. Cấp phát trước (Pre-allocation) & Buffer tái sử dụng
Trong hàm `process()`, thay vì `return output_data` tạo mới, hãy viết đè vào một buffer có sẵn.

```python
# Tốt: Viết đè lên mảng đã cấp phát trước
def process(self, input_data: np.ndarray, output_buffer: np.ndarray):
    np.multiply(input_data, self.gain, out=output_buffer) # Zero-allocation
```

## 5. Tóm tắt Checklist Tối ưu

* [ ] Tính toán DSP đã được bóc tách vào `QThread` chưa?
* [ ] Đã tắt `AutoRange` trên các trục đồ thị `pyqtgraph` chưa?
* [ ] Đã bật `useOpenGL` và `setDownsampling` cho biểu đồ chưa?
* [ ] Tần số gọi hàm `setData()` để vẽ đồ thị đã được giới hạn bằng Timer (30-60 Hz) chưa?
* [ ] Các block DSP đã dùng kỹ thuật tái sử dụng mảng (`out=buffer`) thay vì tạo mảng mới để tránh lag do Garbage Collector chưa?