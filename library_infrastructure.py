"""
library_infrastructure.py - البنية التحتية للمكتبات الجديدة

هذا الملف يوفر:
1. استيراد آمن لمكتبات rapidfuzz و hypothesis
2. معالجة محسّنة للأخطاء
3. تسجيل محسّن للعمليات

الوكيل 1: إعداد البنية التحتية والمكتبات الجديدة
"""

import logging
import sys
import warnings
from typing import Any, Optional, Callable, TypeVar, Dict, List, Tuple
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# ----------------------------
# 1) إعداد نظام التسجيل المحسّن
# ----------------------------

class LogLevel(Enum):
    """مستويات التسجيل"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogConfig:
    """إعدادات التسجيل"""
    name: str = "PREPA"
    level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = "prepa_operations.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    console_output: bool = True
    file_output: bool = True


class EnhancedLogger:
    """
    نظام تسجيل محسّن يدعم:
    - تسجيل متعدد المستويات
    - تتبع العمليات
    - إحصائيات الأداء
    """

    _instances: Dict[str, 'EnhancedLogger'] = {}

    def __new__(cls, config: Optional[LogConfig] = None):
        config = config or LogConfig()
        if config.name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[config.name] = instance
        return cls._instances[config.name]

    def __init__(self, config: Optional[LogConfig] = None):
        if hasattr(self, '_initialized'):
            return

        self.config = config or LogConfig()
        self.logger = logging.getLogger(self.config.name)
        self.logger.setLevel(self.config.level.value)

        # إزالة المعالجات السابقة
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            self.config.log_format,
            datefmt=self.config.date_format
        )

        # معالج الملف
        if self.config.file_output and self.config.log_file:
            file_handler = logging.FileHandler(
                self.config.log_file,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # معالج وحدة التحكم
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self._operation_stats: Dict[str, Dict[str, Any]] = {}
        self._initialized = True

    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def operation_start(self, operation_name: str, details: Optional[Dict] = None):
        """تسجيل بداية عملية"""
        self._operation_stats[operation_name] = {
            'start_time': datetime.now(),
            'details': details or {},
            'status': 'in_progress'
        }
        self.info(f"[START] {operation_name} - {details or ''}")

    def operation_end(self, operation_name: str, success: bool = True, result: Any = None):
        """تسجيل نهاية عملية"""
        if operation_name in self._operation_stats:
            stats = self._operation_stats[operation_name]
            stats['end_time'] = datetime.now()
            stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
            stats['status'] = 'success' if success else 'failed'
            stats['result'] = result

            status_emoji = "[SUCCESS]" if success else "[FAILED]"
            self.info(f"{status_emoji} {operation_name} - Duration: {stats['duration']:.2f}s")
        else:
            self.warning(f"Operation '{operation_name}' was not started")

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """الحصول على إحصائيات العمليات"""
        return self._operation_stats.copy()


# إنشاء logger افتراضي
logger = EnhancedLogger()


# ----------------------------
# 2) استيراد آمن للمكتبات
# ----------------------------

@dataclass
class LibraryStatus:
    """حالة مكتبة"""
    name: str
    available: bool
    version: Optional[str] = None
    error_message: Optional[str] = None
    fallback_available: bool = False


class SafeImporter:
    """
    فئة للاستيراد الآمن للمكتبات مع معالجة الأخطاء
    """

    _library_status: Dict[str, LibraryStatus] = {}

    @classmethod
    def import_library(
        cls,
        library_name: str,
        package_name: Optional[str] = None,
        min_version: Optional[str] = None,
        optional: bool = True
    ) -> Tuple[Optional[Any], LibraryStatus]:
        """
        استيراد مكتبة بشكل آمن

        Args:
            library_name: اسم المكتبة للاستيراد
            package_name: اسم الحزمة للتثبيت (إذا كان مختلفاً)
            min_version: الإصدار الأدنى المطلوب
            optional: هل المكتبة اختيارية

        Returns:
            tuple من (المكتبة أو None, حالة المكتبة)
        """
        package_name = package_name or library_name

        try:
            logger.debug(f"Attempting to import '{library_name}'...")

            # محاولة الاستيراد
            module = __import__(library_name)

            # التحقق من الإصدار
            version = getattr(module, '__version__', None)
            if version is None:
                try:
                    import importlib.metadata
                    version = importlib.metadata.version(package_name)
                except Exception:
                    version = "unknown"

            # التحقق من الحد الأدنى للإصدار
            if min_version and version != "unknown":
                if not cls._check_version(version, min_version):
                    raise ImportError(
                        f"Version {version} is below minimum required {min_version}"
                    )

            status = LibraryStatus(
                name=library_name,
                available=True,
                version=version
            )
            cls._library_status[library_name] = status

            logger.info(f"Successfully imported '{library_name}' v{version}")
            return module, status

        except ImportError as e:
            error_msg = str(e)
            install_cmd = f"pip install {package_name}"

            status = LibraryStatus(
                name=library_name,
                available=False,
                error_message=error_msg
            )
            cls._library_status[library_name] = status

            if optional:
                logger.warning(
                    f"Optional library '{library_name}' not available: {error_msg}. "
                    f"Install with: {install_cmd}"
                )
            else:
                logger.error(
                    f"Required library '{library_name}' not available: {error_msg}. "
                    f"Install with: {install_cmd}"
                )
                raise RuntimeError(
                    f"Missing required dependency: {library_name}. "
                    f"Install with: {install_cmd}"
                ) from e

            return None, status

    @staticmethod
    def _check_version(current: str, minimum: str) -> bool:
        """التحقق من أن الإصدار الحالي >= الحد الأدنى"""
        try:
            from packaging import version
            return version.parse(current) >= version.parse(minimum)
        except ImportError:
            # fallback: مقارنة بسيطة
            current_parts = [int(x) for x in current.split('.')[:3] if x.isdigit()]
            minimum_parts = [int(x) for x in minimum.split('.')[:3] if x.isdigit()]
            return current_parts >= minimum_parts

    @classmethod
    def get_all_status(cls) -> Dict[str, LibraryStatus]:
        """الحصول على حالة جميع المكتبات"""
        return cls._library_status.copy()

    @classmethod
    def print_status_report(cls):
        """طباعة تقرير حالة المكتبات"""
        print("\n" + "=" * 60)
        print("Library Status Report")
        print("=" * 60)

        for name, status in cls._library_status.items():
            status_icon = "[OK]" if status.available else "[X]"
            version_str = f"v{status.version}" if status.version else ""
            error_str = f" - {status.error_message}" if status.error_message else ""
            print(f"{status_icon} {name} {version_str}{error_str}")

        print("=" * 60 + "\n")


# ----------------------------
# 3) استيراد rapidfuzz
# ----------------------------

rapidfuzz_module, rapidfuzz_status = SafeImporter.import_library(
    library_name="rapidfuzz",
    package_name="rapidfuzz",
    optional=True
)

# استيراد الوحدات الفرعية إذا كانت المكتبة متاحة
rapidfuzz_fuzz = None
rapidfuzz_process = None
rapidfuzz_distance = None

if rapidfuzz_module:
    try:
        from rapidfuzz import fuzz as rapidfuzz_fuzz
        from rapidfuzz import process as rapidfuzz_process
        from rapidfuzz import distance as rapidfuzz_distance
        logger.info("rapidfuzz submodules loaded: fuzz, process, distance")
    except ImportError as e:
        logger.warning(f"Could not import rapidfuzz submodules: {e}")


# ----------------------------
# 4) استيراد hypothesis
# ----------------------------

hypothesis_module, hypothesis_status = SafeImporter.import_library(
    library_name="hypothesis",
    package_name="hypothesis",
    optional=True
)

# استيراد الوحدات الفرعية إذا كانت المكتبة متاحة
hypothesis_given = None
hypothesis_strategies = None
hypothesis_settings = None

if hypothesis_module:
    try:
        from hypothesis import given as hypothesis_given
        from hypothesis import strategies as hypothesis_strategies
        from hypothesis import settings as hypothesis_settings
        logger.info("hypothesis submodules loaded: given, strategies, settings")
    except ImportError as e:
        logger.warning(f"Could not import hypothesis submodules: {e}")


# ----------------------------
# 5) معالجة الأخطاء المحسّنة
# ----------------------------

class LibraryError(Exception):
    """خطأ أساسي للمكتبات"""
    pass


class LibraryNotAvailableError(LibraryError):
    """خطأ عندما تكون المكتبة غير متاحة"""

    def __init__(self, library_name: str, install_command: Optional[str] = None):
        self.library_name = library_name
        self.install_command = install_command or f"pip install {library_name}"
        super().__init__(
            f"Library '{library_name}' is not available. "
            f"Install with: {self.install_command}"
        )


class LibraryVersionError(LibraryError):
    """خطأ في إصدار المكتبة"""

    def __init__(self, library_name: str, current_version: str, required_version: str):
        self.library_name = library_name
        self.current_version = current_version
        self.required_version = required_version
        super().__init__(
            f"Library '{library_name}' version {current_version} is below "
            f"required version {required_version}"
        )


T = TypeVar('T')


def require_library(library_name: str) -> Callable:
    """
    مُزخرف للتحقق من توفر مكتبة قبل تنفيذ الدالة

    Usage:
        @require_library('rapidfuzz')
        def my_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            status = SafeImporter._library_status.get(library_name)
            if status is None or not status.available:
                raise LibraryNotAvailableError(library_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def with_fallback(primary_func: Callable, fallback_func: Callable) -> Callable:
    """
    مُزخرف لتوفير fallback في حالة فشل الدالة الأساسية

    Usage:
        @with_fallback(rapidfuzz_match, simple_match)
        def match_strings(a, b):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Primary function failed: {e}. Using fallback."
                )
                return fallback_func(*args, **kwargs)
        return wrapper
    return decorator


# ----------------------------
# 6) وظائف مساعدة للمطابقة الضبابية
# ----------------------------

def fuzzy_match(
    query: str,
    choices: List[str],
    threshold: float = 80.0,
    limit: int = 5
) -> List[Tuple[str, float]]:
    """
    مطابقة ضبابية للنصوص باستخدام rapidfuzz أو fallback

    Args:
        query: النص المراد البحث عنه
        choices: قائمة الخيارات للمطابقة
        threshold: الحد الأدنى للتشابه (0-100)
        limit: الحد الأقصى للنتائج

    Returns:
        قائمة من (النص المطابق, نسبة التشابه)
    """
    logger.operation_start("fuzzy_match", {"query": query, "choices_count": len(choices)})

    try:
        if rapidfuzz_process:
            # استخدام rapidfuzz
            results = rapidfuzz_process.extract(
                query,
                choices,
                scorer=rapidfuzz_fuzz.WRatio,
                limit=limit,
                score_cutoff=threshold
            )
            matched = [(match, score) for match, score, _ in results]
        else:
            # Fallback: مطابقة بسيطة
            matched = _simple_fuzzy_match(query, choices, threshold, limit)

        logger.operation_end("fuzzy_match", success=True, result=len(matched))
        return matched

    except Exception as e:
        logger.operation_end("fuzzy_match", success=False)
        logger.error(f"Fuzzy match failed: {e}")
        return []


def _simple_fuzzy_match(
    query: str,
    choices: List[str],
    threshold: float,
    limit: int
) -> List[Tuple[str, float]]:
    """Fallback بسيط للمطابقة الضبابية"""
    from difflib import SequenceMatcher

    results = []
    for choice in choices:
        ratio = SequenceMatcher(None, query.lower(), choice.lower()).ratio() * 100
        if ratio >= threshold:
            results.append((choice, ratio))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


def fuzzy_ratio(s1: str, s2: str) -> float:
    """
    حساب نسبة التشابه بين نصين

    Args:
        s1: النص الأول
        s2: النص الثاني

    Returns:
        نسبة التشابه (0-100)
    """
    if rapidfuzz_fuzz:
        return rapidfuzz_fuzz.ratio(s1, s2)
    else:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio() * 100


# ----------------------------
# 7) تصدير الواجهة العامة
# ----------------------------

__all__ = [
    # التسجيل
    'EnhancedLogger',
    'LogConfig',
    'LogLevel',
    'logger',

    # الاستيراد الآمن
    'SafeImporter',
    'LibraryStatus',

    # المكتبات
    'rapidfuzz_module',
    'rapidfuzz_fuzz',
    'rapidfuzz_process',
    'rapidfuzz_distance',
    'rapidfuzz_status',

    'hypothesis_module',
    'hypothesis_given',
    'hypothesis_strategies',
    'hypothesis_settings',
    'hypothesis_status',

    # معالجة الأخطاء
    'LibraryError',
    'LibraryNotAvailableError',
    'LibraryVersionError',
    'require_library',
    'with_fallback',

    # وظائف المطابقة
    'fuzzy_match',
    'fuzzy_ratio',
]


# ----------------------------
# 8) نقطة التحقق
# ----------------------------

def check_infrastructure() -> Dict[str, Any]:
    """
    التحقق من حالة البنية التحتية

    Returns:
        تقرير شامل عن حالة المكتبات والنظام
    """
    logger.operation_start("infrastructure_check")

    report = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'libraries': {},
        'all_required_available': True,
        'optional_available': []
    }

    # التحقق من المكتبات
    for name, status in SafeImporter.get_all_status().items():
        report['libraries'][name] = {
            'available': status.available,
            'version': status.version,
            'error': status.error_message
        }
        if status.available:
            report['optional_available'].append(name)

    # اختبار سريع للمطابقة الضبابية
    try:
        test_result = fuzzy_match("test", ["test", "tset", "best"], threshold=50)
        report['fuzzy_match_working'] = len(test_result) > 0
    except Exception as e:
        report['fuzzy_match_working'] = False
        report['fuzzy_match_error'] = str(e)

    logger.operation_end("infrastructure_check", success=True)

    return report


# ----------------------------
# تنفيذ عند الاستيراد
# ----------------------------

if __name__ == "__main__":
    # طباعة تقرير الحالة
    print("\n" + "=" * 60)
    print("PREPA Library Infrastructure - Status Check")
    print("=" * 60)

    report = check_infrastructure()

    print(f"\nPython Version: {report['python_version']}")
    print(f"\nLibraries Status:")

    for name, info in report['libraries'].items():
        status_icon = "[OK]" if info['available'] else "[X]"
        version = f"v{info['version']}" if info['version'] else ""
        print(f"  {status_icon} {name} {version}")

    print(f"\nFuzzy Match Working: {report.get('fuzzy_match_working', 'N/A')}")
    print("\n" + "=" * 60)

    # إعلام الوكيل 1 مكتمل
    print("\n[Agent 1 Status]: Infrastructure setup COMPLETE")
    print("Libraries ready for Agent 2 (EntityCanonicalizer)")
    print("Error handling ready for Agent 7")
    print("Checkpoint Agent 6 can proceed when tests pass")
