# training — Fine-tune NER **multi-task** (type + assertion) song song 2 model

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

Build lại từ data MỚI (đã sửa convention theo test public + gold thật):
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
