# -*- coding: utf-8 -*-
"""Cấu hình nhãn dùng chung cho run_ner.py và postprocess.py.

Không import torch/transformers -> postprocess chạy được trên máy không có GPU.
"""

# Ánh xạ nhãn VietMed (36 loại) -> 5 loại của cuộc thi.
# None = loại bỏ khỏi output theo format cuộc thi.
LABEL_MAP = {
    "DRUGCHEMICAL":       "THUỐC",
    "DISEASESYMTOM":      "TRIỆU_CHỨNG",   # NHẬP NHẰNG: có thể là CHẨN_ĐOÁN (chưa tách được ở phương án A)
    "DIAGNOSTICS":        "TÊN_XÉT_NGHIỆM",
    "UNITCALIBRATOR":     "KẾT_QUẢ_XÉT_NGHIỆM",
    "ORGAN":              None,
    "TREATMENT":          None,
    "SURGERY":            None,
    "MEDDEVICETECHNIQUE": None,
    "PREVENTIVEMED":      None,
    "AGE":                None,
    "GENDER":             None,
    "DATETIME":           None,
    "LOCATION":           None,
    "FOODDRINK":          None,
    "ORGANIZATION":       None,
    "OCCUPATION":         None,
    "TRANSPORTATION":     None,
    "PERSONALCARE":       None,
}

# Nhóm nhãn "thuốc" — cho phép nối chuỗi tên_thuốc + liều + đường_dùng.
DRUG_GROUP = {"DRUGCHEMICAL", "UNITCALIBRATOR"}

# Ký tự cắt ở HAI ĐẦU của span (dấu câu / khoảng trắng dính vào entity).
# Không cắt ký tự nằm GIỮA (vd 'x-quang', '325-650' giữ nguyên).
STRIP_CHARS = " \t\r\n,:;.·•()[]\"'…-–—"

# Danh sách chặn: từ tiêu đề mục / từ chung không phải concept.
# So khớp CHÍNH XÁC (full text sau khi strip + lower). Không đụng 'khó thở', 'đau bụng'...
BLACKLIST = {
    # tiêu đề mục
    "triệu chứng", "triệu chứng hiện tại", "các triệu chứng hiện tại",
    "đặc điểm triệu chứng", "các triệu chứng", "triệu chứng khi khám",
    "chẩn đoán", "chẩn đoán khác", "các chẩn đoán", "các phát hiện chẩn đoán khác",
    "chẩn đoán hình ảnh", "kết quả chẩn đoán", "các kết quả chẩn đoán khác",
    "đánh giá", "đánh giá tại bệnh viện", "khám", "thăm khám", "khám lâm sàng",
    "xét nghiệm", "kết quả xét nghiệm", "kết quả", "kết quả khám lâm sàng",
    "nhập viện", "lý do nhập viện", "nhập viện:", "vào viện",
    "tiền sử", "tiền sử bệnh", "tiền sử bệnh nội khoa", "bệnh sử", "bệnh sử hiện tại",
    "lịch sử bệnh hiện tại", "tiền sử bệnh hiện tại",
    "thủ thuật", "các thủ thuật", "các thủ thuật đã thực hiện",
    "các thủ thuật thực hiện", "thủ thuật thực hiện",
    # từ chung / rác
    "bệnh nhân", "người bệnh", "bác sĩ", "thuốc", "thuốc trước khi nhập viện",
    "điều trị", "bệnh viện", "bệnh lý", "khoa", "cấp cứu", "mức độ",
    "tình trạng", "diễn biến", "các sự kiện", "sự kiện", "yếu tố nguy cơ",
    "các yếu tố nguy cơ liên quan", "n/a", "na", "không", "có",
    "mãn tính", "các bệnh lý mãn tính",
}
