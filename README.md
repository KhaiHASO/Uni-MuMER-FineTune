# Uni-MuMER: Fine-tuning Thống nhất Đa nhiệm của Mô hình Vision-Language cho Nhận dạng Biểu thức Toán học Viết tay

<p align="center">
    <a href="https://arxiv.org/abs/2505.23566"><img src="https://img.shields.io/badge/📄-Paper-red"></a>
    <a href="https://huggingface.co/collections/phxember/uni-mumer-68bfba4747e9289232f3d89e"><img src="https://img.shields.io/badge/🤗 HuggingFace-Data & Models-green"></a>
</p>

## Mô tả

Chúng tôi giới thiệu Uni-MuMER, một phương pháp fine-tune hoàn toàn mô hình Qwen2.5-VL-3B cho tác vụ HMER mà không thay đổi kiến trúc của nó, hiệu quả trong việc tích hợp kiến thức chuyên ngành vào một framework tổng quát. Phương pháp của chúng tôi tích hợp ba tác vụ dựa trên dữ liệu: Tree-Aware Chain-of-Thought (Tree-CoT) cho lập luận không gian có cấu trúc, Error-Driven Learning (EDL) để giảm nhầm lẫn giữa các ký tự trực quan tương tự, và Symbol Counting (SC) để cải thiện tính nhất quán trong nhận dạng các biểu thức dài.

![Uni-MuMER](./asserts/fig/main_fig.drawio_00.png)

Các thí nghiệm trên dataset CROHME và HME100K cho thấy Uni-MuMER đạt được hiệu suất state-of-the-art mới, vượt qua mô hình chuyên biệt nhẹ tốt nhất SSAN 16.31% và VLM hàng đầu Gemini2.5-flash 24.42% trong thiết lập zero-shot.

![intro](./asserts/fig/CROHME_00.png)

## 📢 Cập nhật

- **2025-09-18**: Công trình này được chấp nhận tại NeurIPS 2025 với danh hiệu Spotlight (688/21575).
- **2025-09-09**: Phát hành dataset ([Uni-MuMER-Data](https://huggingface.co/datasets/phxember/Uni-MuMER-Data) và [valid/test data](https://drive.google.com/drive/folders/1T8a3WxICZVl1NJ99hu9tuuqqNZoxGhXq?usp=sharing)) và mã nguồn training. [Xem phần Training]
- **2025-06-02**: Phát hành trọng số mô hình và script inference.

## 🔧 Hướng dẫn Cài đặt và Chạy Chi tiết

### Yêu cầu hệ thống
- Ubuntu (hoặc Linux tương thích)
- GPU với CUDA (khuyến nghị)
- Conda hoặc Miniconda
- Python 3.8+

### Bước 1: Di chuyển vào thư mục project

```bash
cd /home/khai/Desktop/github/Uni-MuMER
```

### Bước 2: Tạo môi trường conda

```bash
# Tạo môi trường conda mới với Python 3.10
conda create -n unimumer python=3.10 -y

# Kích hoạt môi trường
conda activate unimumer
```

**Lưu ý:** Mỗi lần mở terminal mới, bạn cần kích hoạt lại môi trường:
```bash
conda activate unimumer
```

### Bước 3: Kiểm tra phiên bản CUDA (nếu có GPU)

```bash
# Kiểm tra phiên bản CUDA
nvidia-smi
```

Ghi nhớ phiên bản CUDA (ví dụ: 12.4, 11.8, v.v.) để cài đặt PyTorch phù hợp.

### Bước 4: Cài đặt PyTorch với CUDA (bỏ)

```bash
# Đảm bảo đang ở trong môi trường conda
conda activate unimumer

# Cài PyTorch với CUDA 12.4 (thay đổi theo phiên bản CUDA của bạn)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

**Nếu CUDA phiên bản khác:**
- CUDA 11.8: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y`
- CUDA 12.1: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y`

**Nếu không có GPU:**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### Bước 5: Cài đặt Python dependencies

```bash
# Đảm bảo đang ở trong môi trường conda
conda activate unimumer

# Di chuyển vào thư mục project (nếu chưa ở đó)
cd /home/khai/Desktop/github/Uni-MuMER

# Cài đặt các package cần thiết
pip install -r requirements.txt
```

Quá trình cài đặt có thể mất vài phút. Đợi đến khi hoàn tất.

### Bước 6: Giải nén dataset

```bash
# Đảm bảo đang ở trong thư mục project
cd /home/khai/Desktop/github/Uni-MuMER

# Giải nén file data.zip
unzip data.zip
```

Sau khi giải nén, kiểm tra cấu trúc thư mục:
```bash
ls -la data/
```

Bạn sẽ thấy các thư mục:
```
data/
├── CROHME/
├── CROHME2023/
├── HME100K/
├── Im2LaTeXv2/
├── MathWriting/
└── MNE/
```

### Bước 7: Kiểm tra model đã clone

```bash
# Kiểm tra thư mục model
ls -la Uni-MuMER-Qwen2.5-VL-3B/
```

Bạn sẽ thấy các file như `config.json`, `generation_config.json`, v.v.

## 🏃 Inference (Dự đoán)

**Quan trọng:** Luôn kích hoạt môi trường conda trước khi chạy:
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
```

### Cách 1: Chạy tất cả các dataset (Khuyến nghị)

```bash
# Kích hoạt môi trường
conda activate unimumer

# Di chuyển vào thư mục project
cd /home/khai/Desktop/github/Uni-MuMER

# Chạy inference cho tất cả dataset
bash eval/eval_all.sh -m ./Uni-MuMER-Qwen2.5-VL-3B -b 32768
```

**Với GPU cụ thể:**
```bash
# Chỉ định GPU 0
export CUDA_VISIBLE_DEVICES=0
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_all.sh -m ./Uni-MuMER-Qwen2.5-VL-3B -b 32768

# Hoặc sử dụng nhiều GPU (ví dụ: GPU 0 và 1)
export CUDA_VISIBLE_DEVICES=0,1
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_all.sh -m ./Uni-MuMER-Qwen2.5-VL-3B -b 32768
```

**Nếu gặp lỗi OOM (Out of Memory), giảm batch size:**
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_all.sh -m ./Uni-MuMER-Qwen2.5-VL-3B -b 16384
```

### Cách 2: Chạy từng dataset riêng lẻ

#### Dataset CROHME:
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_crohme.sh -i data/CROHME/prompts -o data/CROHME/results -m ./Uni-MuMER-Qwen2.5-VL-3B -b 32768
```

#### Dataset CROHME2023:
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_crohme2023.sh -i data/CROHME2023/prompts -o data/CROHME2023/results -m ./Uni-MuMER-Qwen2.5-VL-3B -b 32768
```

#### Dataset HME100K:
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_hme100k.sh -i data/HME100K/prompts -o data/HME100K/results -m ./Uni-MuMER-Qwen2.5-VL-3B -b 32768
```

#### Dataset Im2LaTeXv2:
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_im2latexv2.sh -i data/Im2LaTeXv2/prompts -o data/Im2LaTeXv2/results -m ./Uni-MuMER-Qwen2.5-VL-3B -b 32768
```

#### Dataset MathWriting:
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_mathwriting.sh -i data/MathWriting/prompts -o data/MathWriting/results -m ./Uni-MuMER-Qwen2.5-VL-3B -b 32768
```

#### Dataset MNE:
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_MNE.sh -i data/MNE/prompts -o data/MNE/results -m ./Uni-MuMER-Qwen2.5-VL-3B -b 32768
```

### Cách 3: Chạy trực tiếp bằng Python

```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER

# Ví dụ: Chạy inference cho CROHME
python scripts/vllm_infer.py \
    --input-dir data/CROHME/prompts \
    --output-dir data/CROHME/results \
    --model ./Uni-MuMER-Qwen2.5-VL-3B \
    --batch-size 32768
```

**Ví dụ khác - HME100K:**
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER

python scripts/vllm_infer.py \
    --input-dir data/HME100K/prompts \
    --output-dir data/HME100K/results \
    --model ./Uni-MuMER-Qwen2.5-VL-3B \
    --batch-size 32768
```

## 📊 Xem kết quả

Sau khi chạy inference, kết quả sẽ được lưu trong các thư mục tương ứng:

```bash
# Xem kết quả CROHME
cat data/CROHME/results/crohme_2014_results.txt
cat data/CROHME/results/crohme_2016_results.txt
cat data/CROHME/results/crohme_2019_results.txt

# Xem kết quả HME100K
cat data/HME100K/results/hme100k_test_results.txt

# Xem kết quả CROHME2023
cat data/CROHME2023/results/crohme2023_test_results.txt
```

Các file kết quả bao gồm:
- `*_pred.json`: Dự đoán của model
- `*_results.txt`: Kết quả đánh giá (accuracy, edit distance, etc.)

## 🏋️ Training (Nếu cần)

Mã nguồn training phụ thuộc vào [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Cài đặt dependencies cho training:

```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER

# Cài đặt các package cần thiết cho training
pip install -r requirements_training.txt
```

Quá trình cài đặt có thể mất khá lâu (10-30 phút tùy vào tốc độ mạng).

### Chạy training:

```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER

# Chạy training
llamafactory-cli train train/Uni-MuMER-train.yaml
```

## ⚠️ Troubleshooting

### Lỗi "conda: command not found"

Cài đặt conda/miniconda:
```bash
# Tải Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Cài đặt
bash Miniconda3-latest-Linux-x86_64.sh

# Khởi động lại terminal hoặc chạy:
source ~/.bashrc

# Sau đó tạo lại môi trường
conda create -n unimumer python=3.10 -y
```

### Lỗi CUDA không khớp

Kiểm tra và cài lại PyTorch với đúng phiên bản CUDA:
```bash
conda activate unimumer

# Kiểm tra phiên bản CUDA
nvidia-smi

# Gỡ PyTorch cũ (nếu cần)
conda uninstall pytorch torchvision torchaudio -y

# Cài lại với phiên bản CUDA đúng (ví dụ: 12.4)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### Lỗi thiếu module

Cài lại dependencies:
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
pip install -r requirements.txt
```

### Lỗi OOM (Out of Memory)

Giảm batch size:
```bash
# Thử với batch size nhỏ hơn
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_all.sh -m ./Uni-MuMER-Qwen2.5-VL-3B -b 16384

# Hoặc nhỏ hơn nữa
bash eval/eval_all.sh -m ./Uni-MuMER-Qwen2.5-VL-3B -b 8192
```

### Quên kích hoạt môi trường

Luôn nhớ kích hoạt môi trường trước khi chạy:
```bash
conda activate unimumer
cd /home/khai/Desktop/github/Uni-MuMER
bash eval/eval_all.sh -m ./Uni-MuMER-Qwen2.5-VL-3B -b 32768
```

### Lỗi "No such file or directory"

Đảm bảo đang ở đúng thư mục project:
```bash
# Kiểm tra thư mục hiện tại
pwd

# Nếu không đúng, di chuyển đến thư mục project
cd /home/khai/Desktop/github/Uni-MuMER

# Kiểm tra lại
ls -la
```

### Lỗi khi import vllm

Cài lại vllm:
```bash
conda activate unimumer
pip uninstall vllm -y
pip install vllm==0.8.5
```

## ✅ TODO

- [x] Inference code and pretrained models.
- [x] Evaluation code.
- [x] Training code.
- [x] Training data.
- [ ] Preprocess code.

## 🙏 Lời cảm ơn

Cảm ơn các dự án sau:

- [CoMER](https://github.com/Green-Wood/CoMER)
- [PosFormer](https://github.com/SJTU-DeepVisionLab/PosFormer)
- [TAMER](https://github.com/qingzhenduyu/TAMER)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [MathNet](https://github.com/felix-schmitt/MathNet)

## 📝 Trích dẫn

Nếu bạn thấy Uni-MuMER hữu ích cho nghiên cứu của mình, vui lòng trích dẫn bài báo của chúng tôi:

```bibtex
@article{li2025unimumer,
  title = {Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition},
  author = {Li, Yu and Jiang, Jin and Zhu, Jianhua and Peng, Shuai and Wei, Baole and Zhou, Yuxuan and Gao, Liangcai},
  year = {2025},
  journal={arXiv preprint arXiv:2505.23566},
}
```
