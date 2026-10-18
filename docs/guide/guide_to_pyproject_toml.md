# Hướng dẫn toàn diện về `pyproject.toml` trong Python

`pyproject.toml` là tệp cấu hình chuẩn (được định nghĩa theo **PEP 518**, **PEP 621** và **PEP 517**) nhằm thống nhất toàn bộ cấu hình dự án, công cụ build và quản lý thư viện vào một nơi duy nhất. Thay vì rải rác các tệp như `setup.py`, `setup.cfg`, `requirements.txt`, `MANIFEST.in`, `tox.ini` hay `.flake8`, giờ đây bạn có thể gom chúng lại trong `pyproject.toml`.

---

## 1. Cấu trúc cơ bản của `pyproject.toml`

Một tệp `pyproject.toml` chuẩn thường chia thành hai phần chính: **Cấu hình hệ thống build** và **Siêu dữ liệu/Cấu hình package**.

### Ví dụ mẫu hoàn chỉnh:

```toml
[build-system]
requires = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "dsp_framework"
version = "0.1.0"
description = "High-level Python wrapper and utilities for a C++ DSP framework"
readme = "README.md"
requires-python = ">=3.9"
license = { text = "MIT" }
authors = [
    { name = "Tên Của Bạn", email = "email@example.com" }
]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "numpy>=1.22.0",
    "matplotlib>=3.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "mypy>=0.990",
    "black",
]

[tool.setuptools.package-data]
dsp_framework = ["py.typed"]

[tool.mypy]
strict = true
warn_return_any = true
warn_unused_configs = true
```

---

## 2. Giải phẫu các thành phần chính

### A. Hệ thống Build (`[build-system]`)
Bắt buộc phải có để các trình cài đặt (như `pip`, `build`) biết cách biên dịch package của bạn thành dạng wheel (`.whl`) hoặc sdist (`.tar.gz`).
* `requires`: Các công cụ cần thiết để build project (thường là `setuptools`, `hatchling`, hoặc `flit-core`).
* `build-backend`: Điểm vào (entry point) để thực hiện quá trình build.

### B. Siêu dữ liệu chuẩn (`[project]` - Theo PEP 621)
Khu vực này chứa thông tin định danh cho package:
* `name`, `version`, `description`: Tên, phiên bản và mô tả ngắn.
* `dependencies`: Các thư viện cốt lõi mà package của bạn bắt buộc phải có để chạy.
* `optional-dependencies`: Các nhóm phụ thuộc tùy chọn (ví dụ: `dev` cho môi trường phát triển, `test` cho kiểm thử).

### C. Cấu hình cho từng công cụ riêng biệt (`[tool.*]`)
Mỗi công cụ phân tích tĩnh hoặc kiểm tra mã nguồn có thể đọc cấu hình riêng của nó trực tiếp từ tệp này thông qua tiền tố `[tool.<ten_cong_cu>]`:
* Ví dụ trên cấu hình riêng cho `mypy` (`[tool.mypy]`) hoặc `setuptools` (`[tool.setuptools.package-data]`). Các công cụ phổ biến khác cũng dùng cách này bao gồm `pytest`, `black`, `ruff`, `isort`.

---

## 3. Lợi ích khi sử dụng `pyproject.toml`

1. **Chuẩn hóa toàn cầu:** Được Python chính thức công nhận qua nhiều đề xuất PEP, thay thế hoàn toàn các file `setup.py` phức tạp chứa mã thực thi Python không an toàn.
2. **Gọn gàng, dễ quản lý:** Tất cả thiết lập từ thông tin package, danh sách thư viện đến cấu hình linter/formatter đều nằm chung một chỗ.
3. **Tương thích tuyệt vời với công cụ hiện đại:** Các công cụ quản lý môi trường thế hệ mới như **Poetry**, **`uv`**, **PDM**, **Hatch** đều lấy `pyproject.toml` làm gốc để vận hành.

---

## 4. Cách quản lý và cập nhật package với `pyproject.toml`

Tùy vào công cụ bạn chọn để khởi tạo môi trường, lệnh cập nhật sẽ khác nhau:

* **Nếu dùng Poetry:**
  * Thêm package mới: `poetry add <package>`
  * Cập nhật package: `poetry update <package>`
* **Nếu dùng `uv`:**
  * Thêm package mới: `uv add <package>`
  * Cập nhật tất cả/khóa phiên bản: `uv lock --upgrade` và `uv sync`
* **Nếu dùng Pip thuần:**
  * Sau khi chạy `pip install --upgrade <package>`, bạn cần **thủ công** cập nhật lại dải phiên bản trong mục `dependencies = [...]` bên trong `pyproject.toml` nếu muốn ghim phiên bản mới cho lần cài đặt sau.




Để thiết lập môi trường ảo (`venv`) bằng `pyproject.toml`, trước tiên chúng ta cần làm rõ một điểm kỹ thuật quan trọng: **Bản thân tệp `pyproject.toml` chỉ là một tệp văn bản cấu hình** (định nghĩa thông tin dự án và danh sách thư viện theo chuẩn PEP 621), chứ nó không tự động tạo ra thư mục môi trường ảo.

Lệnh `python -m venv` mặc định của Python cũng không tự đọc được `pyproject.toml`. Do đó, để dựng môi trường ảo từ `pyproject.toml`, bạn bắt buộc phải thông qua các **công cụ quản lý hiện đại** (như `uv`, Poetry, hoặc PDM). Các công cụ này sẽ đọc `pyproject.toml`, tự động tạo `venv` và cài đặt các thư viện cần thiết.

Dưới đây là cách setup chi tiết bằng 2 công cụ phổ biến và tối ưu nhất hiện nay:

---

### Cách 1: Sử dụng `uv` (Siêu nhanh, hiện đại - Khuyên dùng)

`uv` là công cụ viết bằng Rust, đọc `pyproject.toml` theo chuẩn PEP và quản lý môi trường ảo cực kỳ nhanh chóng.

1. **Chuẩn bị tệp `pyproject.toml**` (hoặc tạo mới bằng lệnh của `uv`):
```bash
uv init --lib my_project
cd my_project

```


Tệp `pyproject.toml` mẫu sẽ được tạo ra với cấu hình chuẩn. Bạn có thể thêm các thư viện vào mục `dependencies`.
2. **Tự động tạo môi trường ảo và cài đặt thư viện:**
Chỉ cần chạy một lệnh duy nhất, `uv` sẽ tự động tạo thư mục môi trường ảo (`.venv`) và đồng bộ các gói:
```bash
export SKBUILD_WHEEL_CMAKE=false
uv sync
```


* `uv` sẽ tự động phát hiện hoặc tạo thư mục `.venv` trong dự án và cài đặt toàn bộ gói từ `pyproject.toml`.


3. **Kích hoạt môi trường ảo:**
* Trên macOS / Linux:
```bash
source .venv/bin/activate

```


* Trên Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1

```





---

### Cách 2: Sử dụng Poetry

Poetry là công cụ lâu đời và rất mạnh mẽ trong việc quản lý gói và môi trường dựa trên `pyproject.toml`.

1. **Khởi tạo dự án (Poetry sẽ tự sinh `pyproject.toml`):**
```bash
poetry init

```


*(Làm theo các bước hướng dẫn trên terminal để khai báo tên, tác giả, và các dependencies).*
2. **Cài đặt môi trường ảo:**
Poetry sẽ tự động tạo một môi trường ảo riêng biệt (thường được lưu trong thư mục hệ thống của Poetry hoặc trong dự án nếu cấu hình `virtualenvs.in-project = true`) dựa trên `pyproject.toml`:
```bash
poetry install

```


Lệnh này đọc `pyproject.toml`, tự tạo `venv`, cài các gói và sinh ra tệp `poetry.lock` để khóa phiên bản.
3. **Kích hoạt môi trường ảo của Poetry:**
```bash
poetry shell

```


Hoặc chạy trực tiếp lệnh bên trong môi trường mà không cần kích hoạt thủ công:
```bash
poetry run python main.py

```



---

# Cách để uv tự động update pyproject.toml

Nếu bạn muốn khi tải package mới mà pyproject.toml tự động được ghi nhận (giống như poetry add hoặc npm install), bạn phải sử dụng lệnh uv add:

```bash
uv add ten_package

```


Khi bạn chạy uv add:
    uv sẽ tự động tải và cài đặt package vào môi trường.
    Nó sẽ tự động viết thêm package đó vào mục dependencies trong pyproject.toml.
    Nó sẽ cập nhật tệp khóa phiên bản (uv.lock) để đảm bảo tính nhất quán.

Tóm lại:
    Dùng uv pip install ten_package: Chỉ cài vào môi trường, không sửa pyproject.toml.
    Dùng uv add ten_package: Vừa cài vào môi trường, vừa tự động update pyproject.toml.

### Tóm tắt luồng hoạt động

1. **Khai báo:** Bạn viết hoặc cấu hình các thư viện cần dùng vào mục `dependencies` bên trong `pyproject.toml`.
2. **Thực thi:** Bạn gọi công cụ quản lý (`uv sync` hoặc `poetry install`).
3. **Kết quả:** Công cụ tự động dựng thư mục `venv` (hoặc `.venv`), tải và cài đặt toàn bộ các thư viện đã khai báo vào đó mà không cần bạn phải thao tác `pip install` từng gói thủ công.



