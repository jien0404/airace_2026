# training — Fine-tune NER **multi-task** (type + assertion) song song 2 model

## Pipeline v2 hiện hành

Phần còn lại của README mô tả lịch sử model v1. Batch mới dùng dataset từ
[`dataset_factory`](../dataset_factory/README.md) và model hybrid:

- BIO head tìm boundary;
- span head kiểm tra type/abstain khi hai head bất đồng;
- assertion head ở mức entity;
- sampled loss cho token `O`;
- sliding window có overlap, không phụ thuộc cấu trúc ba mục;
- RAW không bị NFC hóa trước khi lấy offset.

### Dựng dataset (chạy trên máy có repo đầy đủ)

`dataset_factory.run` đã đóng băng. Luồng hiện hành:

```bash
# 1. sinh draft synthetic (xem dataset_factory/README.md)
python -m dataset_factory.pilot generate --pilot datasets/pilots/train_trial_3000_a_v1 ...

# 2. gộp synthetic + gt2 + part1 thành corpus mức bản ghi; Part 3 chỉ làm test
python -m training.build_ner_corpus \
  --pilot datasets/pilots/train_trial_3000_a_v1 \
  --out datasets/ner_v1/track_a

# 3. cắt cửa sổ trượt cho model v2
python -m training.build_dataset_v2 \
  --dataset-root datasets/ner_v1 --out-root training/dataset_v2 --tracks A
```

Phân vai split và chính sách assertion nằm trong docstring của
[`build_ner_corpus.py`](build_ner_corpus.py) và `datasets/ner_v1/track_a/manifest.json`.

### Train / infer / chấm (máy GPU chỉ cần `training/` + gói dataset)

```bash
python -m training.train_v2 \
  --data-dir training/dataset_v2/track_a \
  --model xlm-roberta-base \
  --out runs/v2/track_a_xlmr --epochs 4 --fp16

python -m training.predict_v2 \
  --model-dir runs/v2/track_a_xlmr/best \
  --input-dir eval/input_turn2 --out-dir runs/v2/track_a_xlmr/pred_part3

python -m training.score_local \
  --pred runs/v2/track_a_xlmr/pred_part3 --gold-zip eval/gold_part3.zip

python -m normalize.run --in_dir runs/v2/track_a_xlmr/pred_part3 --out_dir output_v2_norm
```

`predict_v2` mặc định áp assertion policy `part3`: loại `isFamily` và loại `isNegated` trên
`CHẨN_ĐOÁN`. Dùng `--assertion-policy legacy` để tái tạo hành vi cũ. Thử threshold riêng mà
không đổi checkpoint:

```bash
python -m training.predict_v2 \
  --model runs/v2/track_a_xlmr/best --input input_turn2 \
  --out result/track_a_assertion_v1 \
  --assertion-policy part3 \
  --historical-threshold 0.50 --negated-threshold 0.75
```

Prediction ZIP cũ không còn probability nên chỉ áp được firewall, không đổi threshold:

```bash
python -m training.assertion_policy \
  --input-zip result/track_h_education_fpfix.zip \
  --out-zip result/track_h_education_fpfix_assertion_firewall.zip
```

Thử nghiệm gộp assertion qua các cửa sổ overlap, giữ nguyên span/type theo cửa sổ có NER
confidence cao nhất:

```bash
python -m training.predict_v2 \
  --model runs/track_h_education_fpfix/best --input input_turn2 \
  --out result/track_h_assertion_window_aggregation \
  --assertion-policy part3 --assertion-aggregation max
```

Thử nghiệm window dài hơn (chỉ dùng nếu checkpoint/encoder đủ sức chứa):

```bash
python -m training.predict_v2 \
  --model runs/track_h_education_fpfix/best --input input_turn2 \
  --out result/track_h_assertion_long_window \
  --max-len 512 --max-words 360 --overlap-words 90 \
  --assertion-policy part3 --assertion-aggregation selected
```

Để tách hai tác động, chạy thêm cấu hình kết hợp `--assertion-aggregation max` trên cùng
window dài. Chỉ so WER + assertion; candidates vẫn để nguyên.

[`score_local.py`](score_local.py) chấm theo **chồng lấn + đúng type**, ranh giới không tính —
đúng cơ chế đã cô lập trên leaderboard. Chọn checkpoint theo `overlap.f1`, đừng theo
`boundary_f1_reference_only`.

`training/` không import `dataset_factory` (dùng [`schema_v2.py`](schema_v2.py)) nên máy train
chỉ cần pull thư mục này.

Ma trận segment/document/hybrid, A/B/C và hai backbone nằm trong
[`experiments/`](../experiments/README.md).

> ⛔ Các lệnh build dùng `public_gold`/`synth_800` bên dưới là lịch sử của dataset cũ, đã chuyển
> vào `data_gen/_archive/generated_runs/old_part1/`. Không train batch mới trước khi generator
> đạt [`business_rules/DATASET_CONTRACT.md`](../business_rules/DATASET_CONTRACT.md) và dry-run
> qua `training.verify_dataset`.

Train **ViHealthBERT-syllable** + **XLM-R base** cho bước NER, **multi-task**:
- head TYPE: 5 loại (TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, THUỐC).
- head ASSERTION: 3 nhãn nhị phân (isNegated/isFamily/isHistorical) — 1 model ra CẢ type
  lẫn assertion → nộp được ngay (candidates để rỗng, làm bước sau).

Code chạy trên **server GPU**; dataset giữ private (zip + upload riêng, KHÔNG commit).

File chính: `model.py` (MultiTaskNER), `train_mtl.py` (train), `predict.py` (inference →
output format thi), `build_ner_dataset.py` (tạo dataset), `train_all.sh` (train song song).

## Vì sao 2 model này
Data của ta trộn Anh-Việt nặng (tên thuốc EN, sig `po bid`), syllable-level, nhiều typo:
- **ViHealthBERT-syllable** (`demdecuong/vihealthbert-base-syllable`): domain y tế VN
  (25M câu health), base 135M. Dùng bản **syllable** nên train ở mức syllable, KHÔNG cần
  VnCoreNLP word-segment.
- **XLM-R base** (`xlm-roberta-base`): subword đa ngữ → nuốt tên thuốc EN + sig tự nhiên,
  robust typo/code-mix, không cần segment.

> Đổi model chỉ cần sửa `--model` (hoặc sửa `train_all.sh`). VietMed-NER phobert-v2 /
> CafeBERT cũng chạy được với script này (CafeBERT cần GPU lớn hơn).

## Dữ liệu — nhãn BIO, config-driven, không phụ thuộc offset_mapping
`build_ner_dataset.py` đổi `train.jsonl` (text + entity char-span) thành cửa sổ BIO
token-level:
- val/test tách Ở MỨC NOTE từ nguồn **LLM** (sát thật hơn); train = LLM còn lại + no-LLM.
- `train_ner.py` align subword THỦ CÔNG (nhãn ở subword đầu, -100 phần còn lại) nên chạy
  được CẢ tokenizer slow (ViHealthBERT/PhoBERT) lẫn fast (XLM-R).

### Windowing SECTION-AWARE (train + infer NHẤT QUÁN — `windowing.py`)
Data có cấu trúc mục ("1." "2." "3."). Cửa sổ được cắt **theo mục**, KHÔNG vắt ngang 2 mục
(giữ ngữ cảnh historical/family/type sạch), gói nguyên dòng ≤ `max_words` (không xé entity).
Mục dài hơn `max_words` -> tách nhiều window; window nối tiếp mang **header mục làm prefix
ngữ cảnh** (đưa vào model nhưng KHÔNG tính loss / KHÔNG decode — `n_prefix` trong dataset).
- **`windowing.make_windows` dùng CHUNG** cho `build_ner_dataset` (train) và `predict` (infer)
  -> nhất quán tuyệt đối. Đã test: gộp token content mọi window == đúng chuỗi token note
  (offset chuẩn, mỗi token decode đúng 1 lần).
- ⚠ **`--max_words` khi predict PHẢI khớp lúc build** (mặc định 110 ở cả hai — đừng đổi lệch).

### (Đã build sẵn trên máy local, chỉ cần upload)
```
training/dataset/
  train.jsonl  validation.jsonl  test.jsonl   # {tokens, ner_tags, assert_mask, assert_tags}
  labels.txt  assertions.txt                   # 11 nhãn BIO + 3 assertion
```

Mẫu lệnh lịch sử (không dùng nguyên trạng cho batch mới):
```bash
python -m training.build_ner_dataset \
    --llm  data_gen/generated/public_gold \   # GOLD thật (100 file test public) -> tách val/test REALISTIC
    --extra data_gen/generated/synth_800 \     # synth LLM (convention đã khớp) -> chỉ vào train
    --oversample 5 \                           # lặp gold ×5 để nó có trọng số cao (chuẩn nhất)
    --out training/dataset --val 0.12 --test 0.12
```
- **Vì sao gold làm `--llm`**: val/test lấy từ 100 file THẬT của test public -> chỉ số dev SÁT
  điểm nộp thật; synth chỉ để tăng volume/đa dạng nên vào train.
- **oversample 5**: 76 gold_train ×5 = 380, cân với 800 synth (~32% trọng số real-data).
  Chỉnh 4-6 tùy muốn nghiêng về real (cao hơn = bám phong cách thật/typo nhiều hơn).

> **Nộp CUỐI (sau khi chọn được epoch/hyperparam tốt)**: build lại `--val 0 --test 0`
> để đưa TẤT CẢ 100 gold + synth vào train, rồi train lại -> tận dụng hết real-data.

## Quy trình trên server GPU

1. **Commit code** (không có dataset — đã .gitignore) lên GitHub, pull về server.
2. **Upload dataset**: nén `training/dataset` thành `dataset.zip` (đã tạo sẵn ở local),
   tải lên server, giải nén vào `training/dataset/`:
   ```bash
   cd <repo>/training && unzip -o dataset.zip           # zip đã chứa dataset/ -> training/dataset/*.jsonl
   ```
3. **Cài môi trường** (khớp CUDA server; cài torch theo hướng dẫn pytorch.org nếu cần):
   ```bash
   pip install -r training/requirements.txt
   ```
4. **Train 2 model — 2 tiến trình độc lập, 2 GPU** (khuyến nghị để theo dõi log riêng):
   ```bash
   cd <repo>
   # terminal 1 — GPU 0
   bash training/train_vihealthbert.sh          # log: runs/vihealthbert/train.log
   # terminal 2 — GPU 1
   bash training/train_xlmr.sh                   # log: runs/xlmr/train.log
   ```
   Tùy chỉnh qua biến môi trường: `GPU=0 EPOCHS=5 BS=32 bash training/train_vihealthbert.sh`.
   Chạy nền: thêm `nohup ... &` (log vẫn ghi qua `tee`), theo dõi `tail -f runs/*/train.log`.

   Hoặc 1 lệnh chạy cả hai nền (GPU0+GPU1): `bash training/train_all.sh`.

## Chạy 1 model lẻ
```bash
python -m training.train_mtl --data_dir training/dataset \
    --model demdecuong/vihealthbert-base-syllable --out runs/vihealthbert --fp16
python -m training.train_mtl --data_dir training/dataset \
    --model xlm-roberta-base --out runs/xlmr --fp16
```
Tham số: `--epochs 4 --bs 16 --lr 3e-5 --max_len 256 --assert_weight 1.0 --fp16`.

## Sinh output nộp bài
```bash
# chọn model tốt hơn (theo NER_F1/ASSERT_F1), chạy trên 100 file input/
python -m training.predict --model_dir runs/xlmr/best \
    --input_dir input --out_dir output
cd output && zip -r ../output.zip . && cd ..     # -> output.zip để nộp vòng 1
```
`predict.py` xuất mỗi file `output/<id>.json` đúng schema thi (text/position/type/
assertions/candidates). **candidates để rỗng** (baseline mạnh: GT rỗng + pred rỗng = 1 điểm;
bước chuẩn hóa ICD/RxNorm làm sau).

## Output train
```
runs/<model>/best/              model + tokenizer + heads.pt + mtl_meta.json
runs/<model>/test_metrics.json  NER_F1 + ASSERT_F1 trên test
```
Log in `classification_report` chi tiết F1 theo từng loại entity.

## FAQ — quyết định thiết kế (đọc trước khi train)

**1. 900 note (100 gold + 800 synth) đã đủ chưa?**
Đủ để ra model NER dùng được, KHÔNG phải nhiều. Fine-tune encoder có sẵn cần ÍT data
(vài trăm→vài nghìn note; ở đây ~24k entity). Ràng buộc thật là **độ nhất quán nhãn +
khớp phân phối** (DE_BAI: "bài toán dữ liệu"), không phải số lượng thô. Muốn tăng: sinh
thêm synth (2-5k, đa dạng persona) — lợi ích giảm dần; 100 gold thật vẫn là phần quý nhất.

**2. Fine-tune model đã pretrained có bị "quên kiến thức cũ" (catastrophic forgetting)?**
KHÔNG lo ở đây. Forgetting chỉ là vấn đề khi train TUẦN TỰ nhiều task rồi cần giữ task cũ.
Ta chỉ có MỘT task (NER của mình) → ta CỐ TÌNH tái dụng encoder, "quên" mục tiêu MLM gốc là
vô hại. Cái ta giữ là **biểu diễn ngữ cảnh** (tiếng Việt, thuật ngữ y khoa, code-mix Anh-Việt).

**3. Kiến thức pretrained có làm "nhiễu" không? Có nên dùng model CHƯA pretrained?**
KHÔNG. Với ~900 note, train từ đầu (from scratch) là bất khả thi — encoder cần HÀNG TỶ token
để học ngôn ngữ. Pretrained là NỀN MÓNG giúp giảm nhu cầu data cả nghìn lần, không phải nhiễu.
Rủi ro thật KHÔNG phải "pretrained làm nhiễu" mà là **overfit vào phong cách synth** → chống
bằng: oversample gold thật, ít epoch (3-5), early-stop theo val gold, LR nhỏ (2-3e-5), clip grad.

**4. Vì sao encoder token-classification chứ không phải LLM sinh?**
DE_BAI: `text_score` = WER cần span khớp CHỮ GỐC. Encoder token-classification cho span+offset
"miễn phí" (cắt đúng ký tự nguồn); LLM sinh tự do dễ trôi chữ → mất điểm WER. Đây cũng là runtime
≤9B, self-host, không API — khớp ràng buộc thi.

**5. Chiến lược data cho điểm public tốt nhất (rồi mới tới private):**
- Real gold (100) = phân phối & typo THẬT của test → quý nhất, oversample.
- Synth (800) = phủ đa dạng bệnh/thuốc/xét nghiệm, convention đã khớp → nền volume.
- val/test = gold thật để chọn model. Nộp cuối: gộp hết vào train, train lại.
- Chỉ **điểm nộp public là chân lý** — dùng nó để chốt, val gold chỉ là tín hiệu sớm.

## Lưu ý
- Model tải từ HuggingFace Hub -> server cần internet (hoặc pre-download rồi trỏ `--model`
  vào path local). Nếu qua proxy: `export https_proxy=...`.
- ViHealthBERT tokenizer có thể là slow -> script tự fallback `use_fast=False`, vẫn train
  bình thường nhờ align thủ công.
- Model này ra **type + assertion** (multi-task). Còn **candidates** (ICD cho CHẨN_ĐOÁN,
  RxNorm cho THUỐC) là bước chuẩn hóa riêng — RxNorm đã có ở `dictionary/vocabulary_download_*`
  (định dạng OMOP: CONCEPT.csv, DRUG_STRENGTH.csv…), ICD-10 VN ở `dictionary/DM ICD10-*.xlsx`.
  Bước này để rỗng candidates trước để nộp thử nghiệm nhanh.
