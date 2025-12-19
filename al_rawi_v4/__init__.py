# -*- coding: utf-8 -*-
"""
نظام الراوي الإصدار 4.0
========================
نظام متقدم لمعالجة السيناريوهات العربية وتحويلها إلى مجموعات بيانات عالية الجودة

المكونات الرئيسية:
    - infrastructure: البنية التحتية والمكتبات
    - entity_canonicalizer: توحيد الكيانات
    - quality_filter: فلترة الجودة
    - context_enricher: إثراء السياق
    - temporal_metadata: الميتاداتا الزمنية
"""

__version__ = "4.0.0"
__author__ = "Al-Rawi Team"

from .infrastructure import (
    RAPIDFUZZ_AVAILABLE,
    HYPOTHESIS_AVAILABLE,
    DIFFLIB_AVAILABLE,
    calculate_similarity,
    get_library_status,
    AlRawiLogger,
    setup_logging,
)

__all__ = [
    "RAPIDFUZZ_AVAILABLE",
    "HYPOTHESIS_AVAILABLE",
    "DIFFLIB_AVAILABLE",
    "calculate_similarity",
    "get_library_status",
    "AlRawiLogger",
    "setup_logging",
]
