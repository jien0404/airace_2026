# External challenge: Bệnh viện 108 Q&A

Bộ này dùng để đánh giá checkpoint trên văn phong hỏi–đáp mới, không đưa tự động vào bất kỳ
luồng train/validation nào.

```bash
export http_proxy=http://10.60.117.113:8080
export https_proxy=http://10.60.117.113:8080
export HTTP_PROXY=http://10.60.117.113:8080
export HTTPS_PROXY=http://10.60.117.113:8080
export NO_PROXY=localhost,127.0.0.1,::1,.viettelpost.vn
export no_proxy=localhost,127.0.0.1,::1,.viettelpost.vn

python -m annotation.external_challenge.benhvien108 crawl \
  --data-dir annotation/data/external_benhvien108_qa_v1 \
  --start-page 1 --end-page 16 --delay 0.4
```

Gán nhãn nháp bằng prompt `compiled` của `annotation.part3_eval`:

```bash
python -m annotation.external_challenge.benhvien108 annotate \
  --data-dir annotation/data/external_benhvien108_qa_v1 \
  --run-dir annotation/data/external_benhvien108_qa_v1/part3_eval_run
```

Mở review:

```bash
python -m annotation.app \
  --data_dir annotation/data/external_benhvien108_qa_v1 --port 5000
```

Những record có `strata=alignment_attention` hoặc `strata=empty_llm_draft` cần ưu tiên kiểm tra.
Nhãn LLM chỉ là draft; chỉ các file đã được người dùng review mới được coi là gold để đánh giá.

## Voting với checkpoint

Không ghi đè draft gốc; `labels/` của output là strict voting, còn các phương án đối chứng nằm ở
`llm_labels/`, `model_labels/` và `relaxed_labels/`:

```bash
CUDA_VISIBLE_DEVICES=1 python -m annotation.external_challenge.vote_checkpoint \
  --data-dir annotation/data/external_benhvien108_qa_v1 \
  --checkpoint runs/part3_n2_v3_v67_split_30e_lr2e5/best \
  --out-dir annotation/data/external_benhvien108_qa_v1_model_vote_n2 \
  --assertion-policy part3

python -m annotation.app \
  --data_dir annotation/data/external_benhvien108_qa_v1_model_vote_n2 --port 5000
```
