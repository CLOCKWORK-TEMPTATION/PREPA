#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
وحدة معالجة الأخطاء والتسجيل - نظام الراوي الإصدار 4.0
=====================================================

تحتوي هذه الوحدة على:
- ArabicLogger: مسجل مخصص للرسائل بالعربية
- ErrorHandler: معالج شامل للأخطاء
- DataValidator: التحقق من صحة البيانات
- SafeWriter: ضمان نجاح عمليات الكتابة
- StatisticsCollector: جمع وتسجيل الإحصائيات

المتطلبات المحققة: 6.1، 6.2، 6.4، 6.5
"""

import os
import json
import logging
import hashlib
import traceback
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable, TypeVar, Union
from pathlib import Path
from functools import wraps
from enum import Enum

# ----------------------------
# 1) تعريف مستويات السجل
# ----------------------------
class LogLevel(Enum):
    """مستويات التسجيل المتاحة"""
    DEBUG = "تصحيح"
    INFO = "معلومات"
    WARNING = "تحذير"
    ERROR = "خطأ"
    CRITICAL = "حرج"


# ----------------------------
# 2) رسائل الخطأ بالعربية
# ----------------------------
ARABIC_ERROR_MESSAGES = {
    # أخطاء الملفات
    "file_not_found": "الملف غير موجود: {path}",
    "file_read_error": "فشل في قراءة الملف: {path}، السبب: {reason}",
    "file_write_error": "فشل في كتابة الملف: {path}، السبب: {reason}",
    "file_permission_error": "لا توجد صلاحيات للوصول للملف: {path}",
    "directory_not_found": "المجلد غير موجود: {path}",
    "directory_create_error": "فشل في إنشاء المجلد: {path}، السبب: {reason}",

    # أخطاء البيانات
    "invalid_data": "بيانات غير صالحة: {details}",
    "empty_data": "البيانات فارغة: {field}",
    "missing_field": "حقل مطلوب مفقود: {field}",
    "invalid_type": "نوع بيانات غير صحيح للحقل '{field}': متوقع {expected}، وجد {actual}",
    "validation_error": "فشل التحقق من صحة البيانات: {details}",

    # أخطاء المعالجة
    "parse_error": "فشل في تحليل المحتوى: {details}",
    "encoding_error": "خطأ في ترميز النص: {details}",
    "processing_error": "خطأ أثناء المعالجة: {details}",
    "timeout_error": "انتهت المهلة الزمنية للعملية: {operation}",

    # أخطاء المكتبات
    "library_import_error": "فشل في استيراد المكتبة: {library}، السبب: {reason}",
    "library_not_available": "المكتبة غير متوفرة: {library}. يمكن المتابعة بدون التحسين المعتمد عليها",
    "api_error": "خطأ في الاتصال بالـ API: {details}",

    # أخطاء عامة
    "unknown_error": "خطأ غير متوقع: {details}",
    "operation_failed": "فشلت العملية: {operation}، السبب: {reason}",
    "critical_error": "خطأ حرج يمنع المتابعة: {details}",
}


# ----------------------------
# 3) رسائل النجاح والإحصائيات بالعربية
# ----------------------------
ARABIC_SUCCESS_MESSAGES = {
    # رسائل الملفات
    "file_read_success": "✓ تم قراءة الملف بنجاح: {path}",
    "file_write_success": "✓ تم كتابة الملف بنجاح: {path} ({size} بايت)",
    "directory_created": "✓ تم إنشاء المجلد: {path}",

    # رسائل المعالجة
    "processing_started": "⟳ بدء معالجة: {item}",
    "processing_completed": "✓ اكتملت معالجة: {item}",
    "stage_completed": "✓ اكتملت المرحلة: {stage}",

    # رسائل الإحصائيات
    "statistics_summary": "═══════ إحصائيات {operation} ═══════",
    "total_items": "إجمالي العناصر: {count}",
    "processed_items": "العناصر المعالجة: {count}",
    "skipped_items": "العناصر المتخطاة: {count}",
    "failed_items": "العناصر الفاشلة: {count}",
    "duration": "المدة الزمنية: {duration}",
}


# ----------------------------
# 4) ArabicLogger - مسجل بالعربية
# ----------------------------
class ArabicLogger:
    """
    مسجل مخصص للرسائل بالعربية.

    يوفر واجهة بسيطة لتسجيل الرسائل بمختلف المستويات
    مع دعم كامل للغة العربية وتنسيق RTL.

    المتطلب 6.1: تسجيل رسائل خطأ واضحة باللغة العربية
    """

    def __init__(
        self,
        name: str = "الراوي",
        log_file: Optional[str] = None,
        console_output: bool = True,
        log_level: LogLevel = LogLevel.INFO
    ):
        """
        تهيئة المسجل.

        Args:
            name: اسم المسجل
            log_file: مسار ملف السجل (اختياري)
            console_output: طباعة السجلات في وحدة التحكم
            log_level: مستوى التسجيل الأدنى
        """
        self.name = name
        self.log_file = log_file
        self.console_output = console_output
        self.log_level = log_level
        self._logs: list[dict[str, Any]] = []

        # إعداد مسجل Python الأساسي
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, log_level.name))

        # تنسيق الرسائل
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def _format_message(self, message_key: str, **kwargs) -> str:
        """تنسيق رسالة من قاموس الرسائل"""
        if message_key in ARABIC_ERROR_MESSAGES:
            template = ARABIC_ERROR_MESSAGES[message_key]
        elif message_key in ARABIC_SUCCESS_MESSAGES:
            template = ARABIC_SUCCESS_MESSAGES[message_key]
        else:
            return message_key.format(**kwargs) if kwargs else message_key

        try:
            return template.format(**kwargs)
        except KeyError:
            return template

    def _log(self, level: LogLevel, message: str, **kwargs):
        """تسجيل رسالة داخلية"""
        formatted = self._format_message(message, **kwargs) if kwargs else message

        # حفظ في السجلات الداخلية
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.name,
            "level_ar": level.value,
            "message": formatted,
            "raw_message": message,
            "kwargs": kwargs
        }
        self._logs.append(log_entry)

        # استخدام المسجل الأساسي
        log_method = getattr(self._logger, level.name.lower())
        log_method(formatted)

    def debug(self, message: str, **kwargs):
        """تسجيل رسالة تصحيح"""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """تسجيل رسالة معلومات"""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """تسجيل رسالة تحذير"""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """تسجيل رسالة خطأ"""
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """تسجيل رسالة حرجة"""
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def success(self, message: str, **kwargs):
        """تسجيل رسالة نجاح (تسجل كـ INFO)"""
        self._log(LogLevel.INFO, message, **kwargs)

    def get_logs(self) -> list[dict[str, Any]]:
        """الحصول على جميع السجلات"""
        return self._logs.copy()

    def get_errors(self) -> list[dict[str, Any]]:
        """الحصول على سجلات الأخطاء فقط"""
        return [log for log in self._logs if log["level"] in ("ERROR", "CRITICAL")]

    def has_errors(self) -> bool:
        """التحقق من وجود أخطاء"""
        return len(self.get_errors()) > 0

    def export_logs(self, path: str) -> bool:
        """تصدير السجلات إلى ملف JSON"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._logs, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False


# المسجل العام الافتراضي
_default_logger: Optional[ArabicLogger] = None


def get_logger(name: str = "الراوي") -> ArabicLogger:
    """الحصول على المسجل الافتراضي أو إنشاء واحد جديد"""
    global _default_logger
    if _default_logger is None:
        _default_logger = ArabicLogger(name=name)
    return _default_logger


def set_logger(logger: ArabicLogger):
    """تعيين المسجل الافتراضي"""
    global _default_logger
    _default_logger = logger


# ----------------------------
# 5) ErrorHandler - معالج الأخطاء
# ----------------------------
@dataclass
class ErrorRecord:
    """سجل خطأ واحد"""
    timestamp: str
    error_type: str
    message: str
    details: Optional[str] = None
    traceback_info: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    recovered: bool = False


class ErrorHandler:
    """
    معالج شامل للأخطاء.

    يوفر:
    - تسجيل الأخطاء بالعربية
    - إمكانية التعافي من الأخطاء
    - تجميع الأخطاء للمراجعة
    - المتابعة عند فشل المكتبات الخارجية

    المتطلب 6.1: تسجيل رسائل خطأ واضحة باللغة العربية
    المتطلب 6.3: المتابعة عند فشل مكتبة خارجية
    """

    def __init__(self, logger: Optional[ArabicLogger] = None):
        """
        تهيئة معالج الأخطاء.

        Args:
            logger: المسجل المستخدم (اختياري)
        """
        self.logger = logger or get_logger()
        self._errors: list[ErrorRecord] = []
        self._warnings: list[ErrorRecord] = []

    def record_error(
        self,
        error_type: str,
        message: str,
        details: Optional[str] = None,
        exception: Optional[Exception] = None,
        context: Optional[dict[str, Any]] = None,
        is_warning: bool = False
    ) -> ErrorRecord:
        """
        تسجيل خطأ جديد.

        Args:
            error_type: نوع الخطأ (من ARABIC_ERROR_MESSAGES)
            message: الرسالة الإضافية
            details: تفاصيل إضافية
            exception: الاستثناء الأصلي (اختياري)
            context: سياق إضافي
            is_warning: هل هو تحذير وليس خطأ

        Returns:
            سجل الخطأ المنشأ
        """
        tb_info = None
        if exception:
            tb_info = traceback.format_exc()

        record = ErrorRecord(
            timestamp=datetime.now().isoformat(),
            error_type=error_type,
            message=message,
            details=details,
            traceback_info=tb_info,
            context=context or {}
        )

        if is_warning:
            self._warnings.append(record)
            self.logger.warning(message)
        else:
            self._errors.append(record)
            self.logger.error(message)

        return record

    def handle_exception(
        self,
        exception: Exception,
        operation: str,
        continue_on_error: bool = False,
        context: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        معالجة استثناء.

        Args:
            exception: الاستثناء
            operation: اسم العملية
            continue_on_error: هل نستمر رغم الخطأ
            context: سياق إضافي

        Returns:
            True إذا تم التعامل مع الخطأ ويمكن المتابعة
        """
        error_type = type(exception).__name__
        message = self.logger._format_message(
            "operation_failed",
            operation=operation,
            reason=str(exception)
        )

        self.record_error(
            error_type=error_type,
            message=message,
            details=str(exception),
            exception=exception,
            context=context
        )

        if continue_on_error:
            self.logger.warning(
                f"المتابعة رغم الخطأ في: {operation}"
            )
            return True

        return False

    def handle_library_error(
        self,
        library_name: str,
        exception: Exception
    ) -> bool:
        """
        معالجة خطأ استيراد مكتبة.

        المتطلب 6.3: المتابعة عند فشل مكتبة خارجية

        Args:
            library_name: اسم المكتبة
            exception: الاستثناء

        Returns:
            True دائماً (نستمر بدون المكتبة)
        """
        message = self.logger._format_message(
            "library_not_available",
            library=library_name
        )

        self.record_error(
            error_type="LibraryImportError",
            message=message,
            details=str(exception),
            exception=exception,
            context={"library": library_name},
            is_warning=True
        )

        # نستمر دائماً عند فشل المكتبات الخارجية
        return True

    def get_errors(self) -> list[ErrorRecord]:
        """الحصول على جميع الأخطاء"""
        return self._errors.copy()

    def get_warnings(self) -> list[ErrorRecord]:
        """الحصول على جميع التحذيرات"""
        return self._warnings.copy()

    def has_critical_errors(self) -> bool:
        """التحقق من وجود أخطاء حرجة"""
        return len(self._errors) > 0

    def get_error_summary(self) -> dict[str, Any]:
        """الحصول على ملخص الأخطاء"""
        return {
            "total_errors": len(self._errors),
            "total_warnings": len(self._warnings),
            "error_types": list(set(e.error_type for e in self._errors)),
            "errors": [asdict(e) for e in self._errors],
            "warnings": [asdict(e) for e in self._warnings]
        }

    def clear(self):
        """مسح جميع الأخطاء والتحذيرات"""
        self._errors.clear()
        self._warnings.clear()


# ----------------------------
# 6) DataValidator - التحقق من البيانات
# ----------------------------
@dataclass
class ValidationResult:
    """نتيجة التحقق من الصحة"""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataValidator:
    """
    التحقق من صحة البيانات قبل المعالجة.

    المتطلب 6.4: التحقق من وجود البيانات المطلوبة قبل المعالجة
    """

    def __init__(self, logger: Optional[ArabicLogger] = None):
        self.logger = logger or get_logger()

    def validate_file_exists(self, path: str) -> ValidationResult:
        """التحقق من وجود ملف"""
        result = ValidationResult(is_valid=True)

        if not path:
            result.is_valid = False
            result.errors.append(
                ARABIC_ERROR_MESSAGES["empty_data"].format(field="مسار الملف")
            )
        elif not os.path.exists(path):
            result.is_valid = False
            result.errors.append(
                ARABIC_ERROR_MESSAGES["file_not_found"].format(path=path)
            )
        elif not os.path.isfile(path):
            result.is_valid = False
            result.errors.append(f"المسار ليس ملفاً: {path}")

        return result

    def validate_directory(self, path: str, create_if_missing: bool = False) -> ValidationResult:
        """التحقق من وجود مجلد"""
        result = ValidationResult(is_valid=True)

        if not path:
            result.is_valid = False
            result.errors.append(
                ARABIC_ERROR_MESSAGES["empty_data"].format(field="مسار المجلد")
            )
            return result

        if not os.path.exists(path):
            if create_if_missing:
                try:
                    os.makedirs(path, exist_ok=True)
                    result.warnings.append(f"تم إنشاء المجلد: {path}")
                except Exception as e:
                    result.is_valid = False
                    result.errors.append(
                        ARABIC_ERROR_MESSAGES["directory_create_error"].format(
                            path=path, reason=str(e)
                        )
                    )
            else:
                result.is_valid = False
                result.errors.append(
                    ARABIC_ERROR_MESSAGES["directory_not_found"].format(path=path)
                )
        elif not os.path.isdir(path):
            result.is_valid = False
            result.errors.append(f"المسار ليس مجلداً: {path}")

        return result

    def validate_required_fields(
        self,
        data: dict[str, Any],
        required_fields: list[str]
    ) -> ValidationResult:
        """التحقق من وجود الحقول المطلوبة"""
        result = ValidationResult(is_valid=True)

        if not data:
            result.is_valid = False
            result.errors.append(
                ARABIC_ERROR_MESSAGES["empty_data"].format(field="البيانات")
            )
            return result

        for field_name in required_fields:
            if field_name not in data:
                result.is_valid = False
                result.errors.append(
                    ARABIC_ERROR_MESSAGES["missing_field"].format(field=field_name)
                )
            elif data[field_name] is None:
                result.warnings.append(f"الحقل '{field_name}' فارغ (None)")

        return result

    def validate_type(
        self,
        value: Any,
        expected_type: type,
        field_name: str
    ) -> ValidationResult:
        """التحقق من نوع القيمة"""
        result = ValidationResult(is_valid=True)

        if not isinstance(value, expected_type):
            result.is_valid = False
            result.errors.append(
                ARABIC_ERROR_MESSAGES["invalid_type"].format(
                    field=field_name,
                    expected=expected_type.__name__,
                    actual=type(value).__name__
                )
            )

        return result

    def validate_list_not_empty(
        self,
        data: list,
        field_name: str
    ) -> ValidationResult:
        """التحقق من أن القائمة ليست فارغة"""
        result = ValidationResult(is_valid=True)

        if not isinstance(data, list):
            result.is_valid = False
            result.errors.append(
                ARABIC_ERROR_MESSAGES["invalid_type"].format(
                    field=field_name,
                    expected="list",
                    actual=type(data).__name__
                )
            )
        elif len(data) == 0:
            result.is_valid = False
            result.errors.append(
                ARABIC_ERROR_MESSAGES["empty_data"].format(field=field_name)
            )

        return result

    def validate_scenes(self, scenes: list) -> ValidationResult:
        """التحقق من صحة قائمة المشاهد"""
        result = ValidationResult(is_valid=True)

        # التحقق من أن القائمة ليست فارغة
        list_validation = self.validate_list_not_empty(scenes, "المشاهد")
        if not list_validation.is_valid:
            return list_validation

        # التحقق من كل مشهد
        for i, scene in enumerate(scenes):
            if not hasattr(scene, 'scene_id'):
                result.errors.append(f"المشهد رقم {i+1} لا يحتوي على scene_id")
                result.is_valid = False

            if hasattr(scene, 'dialogue'):
                if not isinstance(scene.dialogue, list):
                    result.errors.append(f"المشهد {getattr(scene, 'scene_id', i+1)}: الحوارات ليست قائمة")
                    result.is_valid = False

        if result.is_valid:
            result.warnings.append(f"تم التحقق من {len(scenes)} مشهد بنجاح")

        return result

    def validate_elements(self, elements: list) -> ValidationResult:
        """التحقق من صحة قائمة العناصر"""
        result = ValidationResult(is_valid=True)

        list_validation = self.validate_list_not_empty(elements, "العناصر")
        if not list_validation.is_valid:
            return list_validation

        for i, element in enumerate(elements):
            if not isinstance(element, dict):
                result.errors.append(f"العنصر رقم {i+1} ليس قاموساً")
                result.is_valid = False
            elif "text" not in element:
                result.warnings.append(f"العنصر رقم {i+1} لا يحتوي على حقل 'text'")

        return result


# ----------------------------
# 7) SafeWriter - كتابة آمنة للملفات
# ----------------------------
@dataclass
class WriteResult:
    """نتيجة عملية الكتابة"""
    success: bool
    path: str
    bytes_written: int = 0
    checksum: Optional[str] = None
    error: Optional[str] = None
    backup_path: Optional[str] = None


class SafeWriter:
    """
    كتابة آمنة للملفات مع التحقق من النجاح.

    المتطلب 6.5: التأكد من نجاح عملية الكتابة قبل المتابعة
    """

    def __init__(
        self,
        logger: Optional[ArabicLogger] = None,
        create_backup: bool = False,
        verify_write: bool = True
    ):
        """
        تهيئة الكاتب الآمن.

        Args:
            logger: المسجل
            create_backup: إنشاء نسخة احتياطية للملفات الموجودة
            verify_write: التحقق من نجاح الكتابة
        """
        self.logger = logger or get_logger()
        self.create_backup = create_backup
        self.verify_write = verify_write
        self._write_results: list[WriteResult] = []

    def _calculate_checksum(self, content: Union[str, bytes]) -> str:
        """حساب checksum للمحتوى"""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()[:16]

    def _create_backup(self, path: str) -> Optional[str]:
        """إنشاء نسخة احتياطية"""
        if not os.path.exists(path):
            return None

        backup_path = f"{path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            import shutil
            shutil.copy2(path, backup_path)
            return backup_path
        except Exception:
            return None

    def write_text(self, path: str, content: str, encoding: str = 'utf-8') -> WriteResult:
        """
        كتابة نص إلى ملف بشكل آمن.

        Args:
            path: مسار الملف
            content: المحتوى النصي
            encoding: ترميز الملف

        Returns:
            نتيجة الكتابة
        """
        result = WriteResult(success=False, path=path)

        try:
            # إنشاء المجلد إذا لم يكن موجوداً
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            # نسخة احتياطية
            if self.create_backup:
                result.backup_path = self._create_backup(path)

            # حساب checksum قبل الكتابة
            expected_checksum = self._calculate_checksum(content)

            # الكتابة
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)

            # التحقق من الكتابة
            if self.verify_write:
                with open(path, 'r', encoding=encoding) as f:
                    written_content = f.read()
                actual_checksum = self._calculate_checksum(written_content)

                if actual_checksum != expected_checksum:
                    result.error = "فشل التحقق: المحتوى المكتوب لا يطابق المحتوى المطلوب"
                    self.logger.error("file_write_error", path=path, reason=result.error)
                    self._write_results.append(result)
                    return result

            # نجاح
            result.success = True
            result.bytes_written = len(content.encode(encoding))
            result.checksum = expected_checksum

            self.logger.success("file_write_success", path=path, size=result.bytes_written)

        except PermissionError:
            result.error = ARABIC_ERROR_MESSAGES["file_permission_error"].format(path=path)
            self.logger.error(result.error)
        except Exception as e:
            result.error = ARABIC_ERROR_MESSAGES["file_write_error"].format(
                path=path, reason=str(e)
            )
            self.logger.error(result.error)

        self._write_results.append(result)
        return result

    def write_json(
        self,
        path: str,
        data: Any,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> WriteResult:
        """
        كتابة JSON إلى ملف بشكل آمن.
        """
        try:
            content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
            return self.write_text(path, content)
        except Exception as e:
            result = WriteResult(
                success=False,
                path=path,
                error=f"فشل في تحويل البيانات إلى JSON: {str(e)}"
            )
            self.logger.error(result.error)
            self._write_results.append(result)
            return result

    def write_jsonl(self, path: str, rows: list[dict[str, Any]]) -> WriteResult:
        """
        كتابة JSONL إلى ملف بشكل آمن.
        """
        try:
            lines = [json.dumps(row, ensure_ascii=False) for row in rows]
            content = "\n".join(lines) + "\n" if lines else ""
            return self.write_text(path, content)
        except Exception as e:
            result = WriteResult(
                success=False,
                path=path,
                error=f"فشل في كتابة JSONL: {str(e)}"
            )
            self.logger.error(result.error)
            self._write_results.append(result)
            return result

    def get_results(self) -> list[WriteResult]:
        """الحصول على جميع نتائج الكتابة"""
        return self._write_results.copy()

    def get_failed_writes(self) -> list[WriteResult]:
        """الحصول على عمليات الكتابة الفاشلة"""
        return [r for r in self._write_results if not r.success]

    def all_successful(self) -> bool:
        """التحقق من نجاح جميع عمليات الكتابة"""
        return all(r.success for r in self._write_results)

    def get_summary(self) -> dict[str, Any]:
        """الحصول على ملخص عمليات الكتابة"""
        return {
            "total_writes": len(self._write_results),
            "successful": len([r for r in self._write_results if r.success]),
            "failed": len([r for r in self._write_results if not r.success]),
            "total_bytes": sum(r.bytes_written for r in self._write_results),
            "failed_paths": [r.path for r in self._write_results if not r.success]
        }


# ----------------------------
# 8) StatisticsCollector - جمع الإحصائيات
# ----------------------------
@dataclass
class OperationStatistics:
    """إحصائيات عملية واحدة"""
    operation_name: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    items_processed: int = 0
    items_skipped: int = 0
    items_failed: int = 0
    details: dict[str, Any] = field(default_factory=dict)


class StatisticsCollector:
    """
    جمع وتسجيل الإحصائيات.

    المتطلب 6.2: تسجيل إحصائيات العمليات المنجزة
    """

    def __init__(self, logger: Optional[ArabicLogger] = None):
        self.logger = logger or get_logger()
        self._operations: dict[str, OperationStatistics] = {}
        self._global_stats: dict[str, Any] = {}

    def start_operation(self, name: str) -> OperationStatistics:
        """بدء تتبع عملية"""
        stats = OperationStatistics(
            operation_name=name,
            start_time=datetime.now().isoformat()
        )
        self._operations[name] = stats
        self.logger.info("processing_started", item=name)
        return stats

    def end_operation(
        self,
        name: str,
        items_processed: int = 0,
        items_skipped: int = 0,
        items_failed: int = 0,
        details: Optional[dict[str, Any]] = None
    ) -> Optional[OperationStatistics]:
        """إنهاء تتبع عملية"""
        if name not in self._operations:
            return None

        stats = self._operations[name]
        stats.end_time = datetime.now().isoformat()

        start = datetime.fromisoformat(stats.start_time)
        end = datetime.fromisoformat(stats.end_time)
        stats.duration_seconds = (end - start).total_seconds()

        stats.items_processed = items_processed
        stats.items_skipped = items_skipped
        stats.items_failed = items_failed
        stats.details = details or {}

        self.logger.info("processing_completed", item=name)
        self._log_statistics(stats)

        return stats

    def _log_statistics(self, stats: OperationStatistics):
        """تسجيل إحصائيات العملية"""
        self.logger.info("statistics_summary", operation=stats.operation_name)
        self.logger.info("processed_items", count=stats.items_processed)
        if stats.items_skipped > 0:
            self.logger.info("skipped_items", count=stats.items_skipped)
        if stats.items_failed > 0:
            self.logger.warning("failed_items", count=stats.items_failed)
        self.logger.info("duration", duration=f"{stats.duration_seconds:.2f} ثانية")

    def add_global_stat(self, key: str, value: Any):
        """إضافة إحصائية عامة"""
        self._global_stats[key] = value

    def increment_stat(self, key: str, amount: int = 1):
        """زيادة قيمة إحصائية"""
        if key not in self._global_stats:
            self._global_stats[key] = 0
        self._global_stats[key] += amount

    def get_operation_stats(self, name: str) -> Optional[OperationStatistics]:
        """الحصول على إحصائيات عملية معينة"""
        return self._operations.get(name)

    def get_all_stats(self) -> dict[str, Any]:
        """الحصول على جميع الإحصائيات"""
        return {
            "operations": {
                name: asdict(stats) for name, stats in self._operations.items()
            },
            "global": self._global_stats
        }

    def print_summary(self):
        """طباعة ملخص الإحصائيات"""
        self.logger.info("═" * 50)
        self.logger.info("ملخص الإحصائيات الشاملة")
        self.logger.info("═" * 50)

        for name, stats in self._operations.items():
            self.logger.info(f"• {name}:")
            self.logger.info(f"  - معالجة: {stats.items_processed}")
            self.logger.info(f"  - متخطاة: {stats.items_skipped}")
            self.logger.info(f"  - فاشلة: {stats.items_failed}")
            self.logger.info(f"  - المدة: {stats.duration_seconds:.2f} ثانية")

        if self._global_stats:
            self.logger.info("─" * 50)
            self.logger.info("إحصائيات عامة:")
            for key, value in self._global_stats.items():
                self.logger.info(f"  - {key}: {value}")

        self.logger.info("═" * 50)

    def export_stats(self, path: str) -> bool:
        """تصدير الإحصائيات إلى ملف JSON"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.get_all_stats(), f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False


# ----------------------------
# 9) Decorators - مزخرفات مساعدة
# ----------------------------
T = TypeVar('T')


def with_error_handling(
    operation_name: str,
    continue_on_error: bool = False,
    default_return: Any = None
):
    """
    مزخرف لإضافة معالجة الأخطاء تلقائياً.

    Usage:
        @with_error_handling("تحليل المشهد")
        def parse_scene(text):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            logger = get_logger()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "operation_failed",
                    operation=operation_name,
                    reason=str(e)
                )
                if not continue_on_error:
                    raise
                return default_return
        return wrapper
    return decorator


def with_statistics(operation_name: str):
    """
    مزخرف لإضافة جمع الإحصائيات تلقائياً.

    Usage:
        @with_statistics("معالجة المشاهد")
        def process_scenes(scenes):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            collector = StatisticsCollector()
            collector.start_operation(operation_name)
            try:
                result = func(*args, **kwargs)
                items = len(result) if hasattr(result, '__len__') else 1
                collector.end_operation(operation_name, items_processed=items)
                return result
            except Exception as e:
                collector.end_operation(operation_name, items_failed=1)
                raise
        return wrapper
    return decorator


def validate_input(**validations):
    """
    مزخرف للتحقق من صحة المدخلات.

    Usage:
        @validate_input(path=lambda x: os.path.exists(x))
        def process_file(path):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            logger = get_logger()

            # دمج args و kwargs
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, validator in validations.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        error_msg = f"فشل التحقق من صحة المعامل: {param_name}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

            return func(*args, **kwargs)
        return wrapper
    return decorator


# ----------------------------
# 10) Safe Library Import - استيراد آمن للمكتبات
# ----------------------------
def safe_import(library_name: str, package_name: Optional[str] = None):
    """
    استيراد مكتبة بشكل آمن مع معالجة الأخطاء.

    المتطلب 5.2: التعامل مع الاستيراد بطريقة آمنة
    المتطلب 6.3: المتابعة عند فشل مكتبة خارجية

    Args:
        library_name: اسم المكتبة للاستيراد
        package_name: اسم الحزمة (إذا اختلف عن library_name)

    Returns:
        الوحدة المستوردة أو None
    """
    logger = get_logger()
    error_handler = ErrorHandler(logger)

    try:
        import importlib
        module = importlib.import_module(library_name)
        logger.debug(f"تم استيراد المكتبة بنجاح: {library_name}")
        return module
    except ImportError as e:
        error_handler.handle_library_error(library_name, e)
        return None


# ----------------------------
# 11) دالة المساعدة للتهيئة
# ----------------------------
def setup_error_handling(
    log_file: Optional[str] = None,
    console_output: bool = True,
    log_level: LogLevel = LogLevel.INFO
) -> tuple[ArabicLogger, ErrorHandler, DataValidator, SafeWriter, StatisticsCollector]:
    """
    تهيئة جميع مكونات معالجة الأخطاء والتسجيل.

    Returns:
        tuple من (logger, error_handler, validator, writer, stats_collector)
    """
    logger = ArabicLogger(
        log_file=log_file,
        console_output=console_output,
        log_level=log_level
    )
    set_logger(logger)

    error_handler = ErrorHandler(logger)
    validator = DataValidator(logger)
    writer = SafeWriter(logger)
    stats_collector = StatisticsCollector(logger)

    return logger, error_handler, validator, writer, stats_collector


# ----------------------------
# للتصدير
# ----------------------------
__all__ = [
    # Enums and Constants
    'LogLevel',
    'ARABIC_ERROR_MESSAGES',
    'ARABIC_SUCCESS_MESSAGES',

    # Classes
    'ArabicLogger',
    'ErrorHandler',
    'ErrorRecord',
    'DataValidator',
    'ValidationResult',
    'SafeWriter',
    'WriteResult',
    'StatisticsCollector',
    'OperationStatistics',

    # Functions
    'get_logger',
    'set_logger',
    'safe_import',
    'setup_error_handling',

    # Decorators
    'with_error_handling',
    'with_statistics',
    'validate_input',
]
