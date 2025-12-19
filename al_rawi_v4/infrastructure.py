# -*- coding: utf-8 -*-
"""
وحدة البنية التحتية والمكتبات الجديدة - نظام الراوي v4.0
=========================================================

هذه الوحدة مسؤولة عن:
    1. الاستيراد الآمن لمكتبات rapidfuzz و hypothesis
    2. معالجة الأخطاء للمكتبات الجديدة
    3. التسجيل المحسّن للعمليات

المتطلبات المحققة: 5.2, 6.1, 6.3
"""

import logging
import sys
import functools
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import difflib


# ============================================
# الجزء 1: الاستيراد الآمن للمكتبات
# ============================================

# متغيرات توفر المكتبات
RAPIDFUZZ_AVAILABLE: bool = False
HYPOTHESIS_AVAILABLE: bool = False
DIFFLIB_AVAILABLE: bool = True  # difflib جزء من المكتبة القياسية

# كائنات المكتبات (تُملأ عند النجاح)
_rapidfuzz_fuzz = None
_rapidfuzz_process = None
_hypothesis_given = None
_hypothesis_strategies = None


def _safe_import_rapidfuzz() -> Tuple[bool, Optional[str]]:
    """
    استيراد آمن لمكتبة rapidfuzz

    Returns:
        tuple: (نجاح الاستيراد, رسالة الخطأ إن وجدت)
    """
    global RAPIDFUZZ_AVAILABLE, _rapidfuzz_fuzz, _rapidfuzz_process

    try:
        from rapidfuzz import fuzz, process
        _rapidfuzz_fuzz = fuzz
        _rapidfuzz_process = process
        RAPIDFUZZ_AVAILABLE = True
        return True, None
    except ImportError as e:
        error_msg = f"فشل استيراد rapidfuzz: {str(e)}"
        return False, error_msg
    except Exception as e:
        error_msg = f"خطأ غير متوقع عند استيراد rapidfuzz: {str(e)}"
        return False, error_msg


def _safe_import_hypothesis() -> Tuple[bool, Optional[str]]:
    """
    استيراد آمن لمكتبة hypothesis

    Returns:
        tuple: (نجاح الاستيراد, رسالة الخطأ إن وجدت)
    """
    global HYPOTHESIS_AVAILABLE, _hypothesis_given, _hypothesis_strategies

    try:
        from hypothesis import given, settings, strategies
        _hypothesis_given = given
        _hypothesis_strategies = strategies
        HYPOTHESIS_AVAILABLE = True
        return True, None
    except ImportError as e:
        error_msg = f"فشل استيراد hypothesis: {str(e)}"
        return False, error_msg
    except Exception as e:
        error_msg = f"خطأ غير متوقع عند استيراد hypothesis: {str(e)}"
        return False, error_msg


# تنفيذ الاستيراد الآمن عند تحميل الوحدة
_rapidfuzz_success, _rapidfuzz_error = _safe_import_rapidfuzz()
_hypothesis_success, _hypothesis_error = _safe_import_hypothesis()


# ============================================
# الجزء 2: واجهات الوصول للمكتبات
# ============================================

def get_rapidfuzz():
    """
    الحصول على وحدة rapidfuzz.fuzz إذا كانت متوفرة

    Returns:
        module أو None
    """
    return _rapidfuzz_fuzz


def get_rapidfuzz_process():
    """
    الحصول على وحدة rapidfuzz.process إذا كانت متوفرة

    Returns:
        module أو None
    """
    return _rapidfuzz_process


def get_hypothesis_given():
    """
    الحصول على decorator hypothesis.given إذا كان متوفراً

    Returns:
        callable أو None
    """
    return _hypothesis_given


def get_hypothesis_strategies():
    """
    الحصول على وحدة hypothesis.strategies إذا كانت متوفرة

    Returns:
        module أو None
    """
    return _hypothesis_strategies


# ============================================
# الجزء 3: دوال حساب التشابه
# ============================================

def calculate_similarity(s1: str, s2: str) -> float:
    """
    حساب نسبة التشابه بين نصين باستخدام أفضل مكتبة متوفرة

    تستخدم rapidfuzz إذا كانت متوفرة (أسرع)، وإلا تستخدم difflib

    Args:
        s1: النص الأول
        s2: النص الثاني

    Returns:
        نسبة التشابه (0.0 إلى 1.0)
    """
    if not s1 or not s2:
        return 0.0

    if s1 == s2:
        return 1.0

    if RAPIDFUZZ_AVAILABLE and _rapidfuzz_fuzz is not None:
        # rapidfuzz تعيد نسبة من 0-100، نحولها إلى 0-1
        return _rapidfuzz_fuzz.ratio(s1, s2) / 100.0
    else:
        # استخدام difflib كبديل
        return difflib.SequenceMatcher(None, s1, s2).ratio()


def calculate_similarity_batch(query: str, choices: List[str], threshold: float = 0.85) -> List[Tuple[str, float]]:
    """
    حساب التشابه بين نص واحد ومجموعة من النصوص

    Args:
        query: النص المرجعي
        choices: قائمة النصوص للمقارنة
        threshold: الحد الأدنى للتشابه

    Returns:
        قائمة من الأزواج (النص، نسبة التشابه) للنصوص التي تجاوزت الحد
    """
    results = []

    if RAPIDFUZZ_AVAILABLE and _rapidfuzz_process is not None:
        # استخدام rapidfuzz.process للأداء الأفضل
        matches = _rapidfuzz_process.extract(query, choices, limit=None)
        for choice, score, _ in matches:
            similarity = score / 100.0
            if similarity >= threshold:
                results.append((choice, similarity))
    else:
        # استخدام difflib
        for choice in choices:
            similarity = calculate_similarity(query, choice)
            if similarity >= threshold:
                results.append((choice, similarity))

    return sorted(results, key=lambda x: x[1], reverse=True)


# ============================================
# الجزء 4: حالة المكتبات والتشخيص
# ============================================

class LibraryStatus(Enum):
    """حالة توفر المكتبة"""
    AVAILABLE = "متوفرة"
    NOT_INSTALLED = "غير مثبتة"
    IMPORT_ERROR = "خطأ في الاستيراد"


@dataclass
class LibraryInfo:
    """معلومات عن مكتبة معينة"""
    name: str
    status: LibraryStatus
    version: Optional[str] = None
    error_message: Optional[str] = None
    fallback_available: bool = False
    fallback_name: Optional[str] = None


def get_library_status() -> Dict[str, LibraryInfo]:
    """
    الحصول على حالة جميع المكتبات المطلوبة

    Returns:
        قاموس بمعلومات كل مكتبة
    """
    status = {}

    # rapidfuzz
    if RAPIDFUZZ_AVAILABLE:
        try:
            import rapidfuzz
            version = getattr(rapidfuzz, "__version__", "غير معروف")
        except Exception:
            version = "غير معروف"
        status["rapidfuzz"] = LibraryInfo(
            name="rapidfuzz",
            status=LibraryStatus.AVAILABLE,
            version=version,
            fallback_available=True,
            fallback_name="difflib"
        )
    else:
        status["rapidfuzz"] = LibraryInfo(
            name="rapidfuzz",
            status=LibraryStatus.NOT_INSTALLED,
            error_message=_rapidfuzz_error,
            fallback_available=True,
            fallback_name="difflib"
        )

    # hypothesis
    if HYPOTHESIS_AVAILABLE:
        try:
            import hypothesis
            version = getattr(hypothesis, "__version__", "غير معروف")
        except Exception:
            version = "غير معروف"
        status["hypothesis"] = LibraryInfo(
            name="hypothesis",
            status=LibraryStatus.AVAILABLE,
            version=version,
            fallback_available=False
        )
    else:
        status["hypothesis"] = LibraryInfo(
            name="hypothesis",
            status=LibraryStatus.NOT_INSTALLED,
            error_message=_hypothesis_error,
            fallback_available=False
        )

    # difflib (دائماً متوفرة)
    status["difflib"] = LibraryInfo(
        name="difflib",
        status=LibraryStatus.AVAILABLE,
        version=sys.version.split()[0],  # إصدار بايثون
        fallback_available=False
    )

    return status


def print_library_status():
    """طباعة حالة المكتبات بتنسيق واضح"""
    status = get_library_status()
    print("\n" + "=" * 60)
    print("حالة المكتبات - نظام الراوي v4.0")
    print("=" * 60)

    for name, info in status.items():
        status_icon = "✓" if info.status == LibraryStatus.AVAILABLE else "✗"
        print(f"\n{status_icon} {info.name}:")
        print(f"  الحالة: {info.status.value}")

        if info.version:
            print(f"  الإصدار: {info.version}")

        if info.error_message:
            print(f"  الخطأ: {info.error_message}")

        if info.fallback_available:
            print(f"  البديل: {info.fallback_name}")

    print("\n" + "=" * 60)


# ============================================
# الجزء 5: نظام التسجيل المحسّن
# ============================================

class AlRawiLogLevel(Enum):
    """مستويات التسجيل"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogEntry:
    """سجل واحد"""
    timestamp: datetime
    level: str
    module: str
    message: str
    extra_data: Dict[str, Any] = field(default_factory=dict)


class AlRawiLogger:
    """
    نظام تسجيل محسّن لنظام الراوي

    يوفر:
        - تسجيل بالعربية
        - تتبع الإحصائيات
        - تصدير السجلات
    """

    def __init__(
        self,
        name: str = "al_rawi",
        log_file: Optional[str] = None,
        level: AlRawiLogLevel = AlRawiLogLevel.INFO,
        console_output: bool = True
    ):
        """
        تهيئة المسجل

        Args:
            name: اسم المسجل
            log_file: مسار ملف السجل (اختياري)
            level: مستوى التسجيل
            console_output: إظهار السجلات في الكونسول
        """
        self.name = name
        self.log_entries: List[LogEntry] = []
        self.stats: Dict[str, int] = {
            "debug": 0,
            "info": 0,
            "warning": 0,
            "error": 0,
            "critical": 0
        }

        # إعداد المسجل الأساسي
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)
        self._logger.handlers.clear()

        # تنسيق الرسائل
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # إضافة handler للكونسول
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # إضافة handler للملف
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def _log(self, level: str, message: str, **kwargs):
        """تسجيل رسالة داخلياً"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            module=self.name,
            message=message,
            extra_data=kwargs
        )
        self.log_entries.append(entry)
        self.stats[level.lower()] += 1

    def debug(self, message: str, **kwargs):
        """تسجيل رسالة تصحيح"""
        self._log("DEBUG", message, **kwargs)
        self._logger.debug(message)

    def info(self, message: str, **kwargs):
        """تسجيل رسالة معلومات"""
        self._log("INFO", message, **kwargs)
        self._logger.info(message)

    def warning(self, message: str, **kwargs):
        """تسجيل تحذير"""
        self._log("WARNING", message, **kwargs)
        self._logger.warning(message)

    def error(self, message: str, **kwargs):
        """تسجيل خطأ"""
        self._log("ERROR", message, **kwargs)
        self._logger.error(message)

    def critical(self, message: str, **kwargs):
        """تسجيل خطأ حرج"""
        self._log("CRITICAL", message, **kwargs)
        self._logger.critical(message)

    # رسائل محددة للنظام

    def log_library_import(self, library_name: str, success: bool, error: Optional[str] = None):
        """تسجيل نتيجة استيراد مكتبة"""
        if success:
            self.info(f"تم استيراد مكتبة {library_name} بنجاح")
        else:
            self.warning(f"فشل استيراد مكتبة {library_name}: {error or 'سبب غير معروف'}")

    def log_operation_start(self, operation_name: str, **details):
        """تسجيل بداية عملية"""
        details_str = ", ".join(f"{k}={v}" for k, v in details.items()) if details else ""
        self.info(f"بدء عملية: {operation_name}" + (f" ({details_str})" if details_str else ""))

    def log_operation_end(self, operation_name: str, success: bool = True, **results):
        """تسجيل نهاية عملية"""
        results_str = ", ".join(f"{k}={v}" for k, v in results.items()) if results else ""
        status = "اكتملت" if success else "فشلت"
        self.info(f"{status} عملية: {operation_name}" + (f" ({results_str})" if results_str else ""))

    def log_statistics(self, stats: Dict[str, Any]):
        """تسجيل إحصائيات"""
        self.info("إحصائيات العملية:")
        for key, value in stats.items():
            self.info(f"  - {key}: {value}")

    def log_data_validation(self, field_name: str, valid: bool, value: Any = None):
        """تسجيل نتيجة التحقق من البيانات"""
        if valid:
            self.debug(f"التحقق من {field_name}: ناجح")
        else:
            self.warning(f"التحقق من {field_name}: فشل (القيمة: {value})")

    def log_file_operation(self, operation: str, file_path: str, success: bool = True, error: Optional[str] = None):
        """تسجيل عملية ملف"""
        if success:
            self.info(f"عملية الملف '{operation}' نجحت: {file_path}")
        else:
            self.error(f"عملية الملف '{operation}' فشلت: {file_path} - {error or 'سبب غير معروف'}")

    def get_stats(self) -> Dict[str, int]:
        """الحصول على إحصائيات التسجيل"""
        return self.stats.copy()

    def get_entries(self, level: Optional[str] = None, limit: int = 100) -> List[LogEntry]:
        """
        الحصول على السجلات

        Args:
            level: فلترة حسب المستوى (اختياري)
            limit: الحد الأقصى للسجلات

        Returns:
            قائمة السجلات
        """
        entries = self.log_entries
        if level:
            entries = [e for e in entries if e.level.upper() == level.upper()]
        return entries[-limit:]

    def export_logs(self, file_path: str):
        """تصدير السجلات إلى ملف JSON"""
        import json

        data = {
            "logger_name": self.name,
            "statistics": self.stats,
            "entries": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "level": e.level,
                    "module": e.module,
                    "message": e.message,
                    "extra_data": e.extra_data
                }
                for e in self.log_entries
            ]
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.info(f"تم تصدير السجلات إلى: {file_path}")


# المسجل الافتراضي
_default_logger: Optional[AlRawiLogger] = None


def setup_logging(
    name: str = "al_rawi",
    log_file: Optional[str] = None,
    level: AlRawiLogLevel = AlRawiLogLevel.INFO,
    console_output: bool = True
) -> AlRawiLogger:
    """
    إعداد نظام التسجيل

    Args:
        name: اسم المسجل
        log_file: مسار ملف السجل
        level: مستوى التسجيل
        console_output: إظهار في الكونسول

    Returns:
        كائن AlRawiLogger
    """
    global _default_logger
    _default_logger = AlRawiLogger(
        name=name,
        log_file=log_file,
        level=level,
        console_output=console_output
    )

    # تسجيل حالة المكتبات عند البدء
    _default_logger.log_library_import("rapidfuzz", RAPIDFUZZ_AVAILABLE, _rapidfuzz_error)
    _default_logger.log_library_import("hypothesis", HYPOTHESIS_AVAILABLE, _hypothesis_error)
    _default_logger.log_library_import("difflib", True)

    return _default_logger


def get_logger() -> AlRawiLogger:
    """الحصول على المسجل الافتراضي"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger


# ============================================
# الجزء 6: معالجة الأخطاء المحسّنة
# ============================================

class AlRawiError(Exception):
    """الفئة الأساسية لأخطاء نظام الراوي"""

    def __init__(self, message: str, error_code: str = "UNKNOWN", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class LibraryNotAvailableError(AlRawiError):
    """خطأ عند عدم توفر مكتبة مطلوبة"""

    def __init__(self, library_name: str, fallback_used: bool = False):
        message = f"المكتبة '{library_name}' غير متوفرة"
        if fallback_used:
            message += " - تم استخدام البديل"
        super().__init__(message, "LIB_NOT_AVAILABLE", {"library": library_name, "fallback_used": fallback_used})


class DataValidationError(AlRawiError):
    """خطأ في التحقق من صحة البيانات"""

    def __init__(self, field_name: str, expected: str, actual: Any):
        message = f"فشل التحقق من حقل '{field_name}': المتوقع {expected}، الفعلي {actual}"
        super().__init__(message, "DATA_VALIDATION", {"field": field_name, "expected": expected, "actual": actual})


class FileOperationError(AlRawiError):
    """خطأ في عمليات الملفات"""

    def __init__(self, operation: str, file_path: str, reason: str):
        message = f"فشلت عملية '{operation}' على الملف '{file_path}': {reason}"
        super().__init__(message, "FILE_OPERATION", {"operation": operation, "file_path": file_path, "reason": reason})


def safe_operation(fallback_value: Any = None, log_errors: bool = True):
    """
    Decorator لتنفيذ عملية بشكل آمن مع معالجة الأخطاء

    Args:
        fallback_value: القيمة التي تُرجع عند حدوث خطأ
        log_errors: تسجيل الأخطاء في السجل
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except AlRawiError:
                raise  # نعيد رمي أخطاء النظام كما هي
            except Exception as e:
                logger = get_logger()
                error_msg = f"خطأ في تنفيذ {func.__name__}: {str(e)}"
                if log_errors:
                    logger.error(error_msg, exception=str(e), function=func.__name__)
                if fallback_value is not None:
                    return fallback_value
                raise AlRawiError(error_msg, "EXECUTION_ERROR", {"function": func.__name__, "exception": str(e)})
        return wrapper
    return decorator


def require_library(library_name: str, fallback_allowed: bool = True):
    """
    Decorator للتأكد من توفر مكتبة قبل تنفيذ الدالة

    Args:
        library_name: اسم المكتبة المطلوبة
        fallback_allowed: السماح باستخدام البديل
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            status = get_library_status()
            lib_info = status.get(library_name)

            if lib_info is None:
                raise LibraryNotAvailableError(library_name)

            if lib_info.status != LibraryStatus.AVAILABLE:
                if fallback_allowed and lib_info.fallback_available:
                    logger = get_logger()
                    logger.warning(f"المكتبة '{library_name}' غير متوفرة، سيتم استخدام '{lib_info.fallback_name}'")
                else:
                    raise LibraryNotAvailableError(library_name)

            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================
# الجزء 7: دوال مساعدة
# ============================================

def count_arabic_words(text: str) -> int:
    """
    عدّ الكلمات في نص عربي

    Args:
        text: النص المراد عدّ كلماته

    Returns:
        عدد الكلمات
    """
    if not text:
        return 0
    # تقسيم النص بالمسافات وعدّ العناصر غير الفارغة
    words = [w for w in text.split() if w.strip()]
    return len(words)


def normalize_arabic_text(text: str) -> str:
    """
    تطبيع النص العربي (إزالة التشكيل والتوحيد)

    Args:
        text: النص المراد تطبيعه

    Returns:
        النص المطبّع
    """
    import unicodedata
    import re

    if not text:
        return ""

    # إزالة التشكيل
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

    # توحيد الهمزات
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)

    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def validate_file_exists(file_path: str) -> bool:
    """
    التحقق من وجود ملف

    Args:
        file_path: مسار الملف

    Returns:
        True إذا كان الملف موجوداً
    """
    return Path(file_path).exists()


def ensure_directory(dir_path: str) -> Path:
    """
    التأكد من وجود مجلد (إنشاؤه إذا لم يكن موجوداً)

    Args:
        dir_path: مسار المجلد

    Returns:
        كائن Path للمجلد
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================
# الجزء 8: تهيئة الوحدة
# ============================================

def _initialize_module():
    """تهيئة الوحدة عند التحميل"""
    # لا نسجل هنا لتجنب التسجيل المبكر
    # سيتم التسجيل عند استدعاء setup_logging()
    pass


# تنفيذ التهيئة
_initialize_module()


# ============================================
# واجهة للتصدير
# ============================================

__all__ = [
    # ثوابت توفر المكتبات
    "RAPIDFUZZ_AVAILABLE",
    "HYPOTHESIS_AVAILABLE",
    "DIFFLIB_AVAILABLE",

    # دوال الوصول للمكتبات
    "get_rapidfuzz",
    "get_rapidfuzz_process",
    "get_hypothesis_given",
    "get_hypothesis_strategies",

    # دوال حساب التشابه
    "calculate_similarity",
    "calculate_similarity_batch",

    # حالة المكتبات
    "LibraryStatus",
    "LibraryInfo",
    "get_library_status",
    "print_library_status",

    # نظام التسجيل
    "AlRawiLogLevel",
    "LogEntry",
    "AlRawiLogger",
    "setup_logging",
    "get_logger",

    # معالجة الأخطاء
    "AlRawiError",
    "LibraryNotAvailableError",
    "DataValidationError",
    "FileOperationError",
    "safe_operation",
    "require_library",

    # دوال مساعدة
    "count_arabic_words",
    "normalize_arabic_text",
    "validate_file_exists",
    "ensure_directory",
]
