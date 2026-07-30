"""Schema nhãn cho pipeline v2.

Bản sao độc lập của `dataset_factory.schema` để thư mục `training/` chạy được trên máy chỉ
pull code training (dataset đi kèm dưới dạng zip, không cần `dataset_factory`).
Nếu đổi type/assertion thì phải đổi cả hai nơi.
"""

from __future__ import annotations

TYPES = (
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "THUỐC",
)
ASSERTIONS = ("isNegated", "isFamily", "isHistorical")
ASSERTION_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
CANDIDATE_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}
