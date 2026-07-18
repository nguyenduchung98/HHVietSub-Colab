# HHVietSub Colab

GPU backend riêng cho HHVietSub. Server cung cấp API OmniVoice qua Google Colab và Cloudflare Tunnel.

## Nguồn model

- Source/runtime: https://github.com/k2-fsa/OmniVoice (Apache-2.0)
- Weights: https://huggingface.co/k2-fsa/OmniVoice

## Sử dụng

Mở `HHVietSub_Colab.ipynb` bằng Google Colab, chọn GPU và Run all. Sao chép URL `trycloudflare.com` vào tab Cấu hình của HHVietSub. Token hiển thị trong notebook phải được nhập cùng URL.

Không commit model hoặc audio người dùng vào repository.

