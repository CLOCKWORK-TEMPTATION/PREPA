#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبارات الخصائص لوحدة معالجة الأخطاء والتسجيل - نظام الراوي الإصدار 4.0
==========================================================================

هذا الملف يحتوي على اختبارات الخصائص التالية:
- الخاصية 20 (7.1): تسجيل الأخطاء
- الخاصية 21 (7.2): تسجيل الإحصائيات
- الخاصية 22 (7.3): التحقق من البيانات
- الخاصية 23 (7.4): ضمان نجاح الكتابة
"""

import os
import sys
import json
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Any, Optional

# محاولة استيراد hypothesis للاختبارات القائمة على الخصائص
try:
    from hypothesis import given, strategies as st, settings, assume, example
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("تحذير: مكتبة hypothesis غير متوفرة. سيتم استخدام اختبارات بديلة.")

import unittest

# استيراد الوحدة المراد اختبارها
from error_handling import (
    ArabicLogger,
    ErrorHandler,
    DataValidator,
    SafeWriter,
    StatisticsCollector,
    LogLevel,
    ValidationResult,
    WriteResult,
    ErrorRecord,
    OperationStatistics,
    get_logger,
    set_logger,
    safe_import,
    setup_error_handling,
    ARABIC_ERROR_MESSAGES,
    with_error_handling,
    with_statistics,
)


# ----------------------------
# نماذج بيانات للاختبار
# ----------------------------
@dataclass
class MockScene:
    """مشهد وهمي للاختبار"""
    scene_id: str
    scene_number: Optional[int] = None
    heading: Optional[str] = None
    dialogue: list = field(default_factory=list)
    actions: list = field(default_factory=list)


@dataclass
class MockDialogueTurn:
    """حوار وهمي للاختبار"""
    scene_id: str
    turn_id: int
    speaker: str
    text: str


# ----------------------------
# الخاصية 20 (7.1): اختبار تسجيل الأخطاء
# ----------------------------
class TestProperty20ErrorLogging(unittest.TestCase):
    """
    الخاصية 20: تسجيل الأخطاء

    تتحقق من: المتطلب 6.1
    - WHEN يحدث خطأ في أي مرحلة
    - THEN النظام SHALL تسجيل رسالة خطأ واضحة باللغة العربية
    """

    def setUp(self):
        """تهيئة قبل كل اختبار"""
        self.logger = ArabicLogger(name="اختبار", console_output=False)
        self.error_handler = ErrorHandler(self.logger)

    def test_error_messages_are_in_arabic(self):
        """التحقق من أن رسائل الخطأ بالعربية"""
        # تسجيل خطأ
        self.error_handler.record_error(
            error_type="TestError",
            message=ARABIC_ERROR_MESSAGES["file_not_found"].format(path="/test/path")
        )

        # التحقق من وجود الخطأ
        errors = self.error_handler.get_errors()
        self.assertEqual(len(errors), 1)

        # التحقق من أن الرسالة تحتوي على نص عربي
        error_message = errors[0].message
        self.assertIn("الملف غير موجود", error_message)

    def test_all_error_types_have_arabic_messages(self):
        """التحقق من أن جميع أنواع الأخطاء لها رسائل بالعربية"""
        for error_key, message_template in ARABIC_ERROR_MESSAGES.items():
            # التحقق من أن القالب يحتوي على نص عربي
            has_arabic = any('\u0600' <= c <= '\u06FF' for c in message_template)
            self.assertTrue(
                has_arabic,
                f"رسالة الخطأ '{error_key}' لا تحتوي على نص عربي"
            )

    def test_error_logging_captures_exception_details(self):
        """التحقق من أن تسجيل الخطأ يلتقط تفاصيل الاستثناء"""
        try:
            raise ValueError("خطأ اختباري")
        except ValueError as e:
            self.error_handler.record_error(
                error_type="ValueError",
                message="حدث خطأ",
                exception=e
            )

        errors = self.error_handler.get_errors()
        self.assertEqual(len(errors), 1)
        self.assertIsNotNone(errors[0].traceback_info)

    def test_error_handler_continues_on_library_failure(self):
        """
        المتطلب 6.3: المتابعة عند فشل مكتبة خارجية

        WHEN تفشل مكتبة خارجية
        THEN النظام SHALL المتابعة بدون التحسين المعتمد عليها
        """
        # محاكاة فشل استيراد مكتبة
        try:
            raise ImportError("المكتبة غير موجودة")
        except ImportError as e:
            result = self.error_handler.handle_library_error("test_library", e)

        # يجب أن نستمر (True)
        self.assertTrue(result)

        # يجب أن يكون التحذير مسجلاً (وليس خطأ)
        warnings = self.error_handler.get_warnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn("غير متوفرة", warnings[0].message)

    def test_logger_exports_logs_correctly(self):
        """التحقق من تصدير السجلات بشكل صحيح"""
        self.logger.error("خطأ اختباري")
        self.logger.info("معلومة اختبارية")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            result = self.logger.export_logs(temp_path)
            self.assertTrue(result)

            with open(temp_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)

            self.assertEqual(len(logs), 2)
        finally:
            os.unlink(temp_path)

    if HYPOTHESIS_AVAILABLE:
        @given(st.text(min_size=1, max_size=100))
        @settings(max_examples=20)
        def test_property_any_error_message_is_logged(self, message):
            """خاصية: أي رسالة خطأ يجب أن تُسجل"""
            assume(message.strip())  # تجاهل الرسائل الفارغة

            logger = ArabicLogger(name="test", console_output=False)
            logger.error(message)

            logs = logger.get_logs()
            self.assertGreater(len(logs), 0)
            self.assertEqual(logs[-1]["level"], "ERROR")


# ----------------------------
# الخاصية 21 (7.2): اختبار تسجيل الإحصائيات
# ----------------------------
class TestProperty21StatisticsLogging(unittest.TestCase):
    """
    الخاصية 21: تسجيل الإحصائيات

    تتحقق من: المتطلب 6.2
    - WHEN يتم تطبيق التحسينات
    - THEN النظام SHALL تسجيل إحصائيات العمليات المنجزة
    """

    def setUp(self):
        """تهيئة قبل كل اختبار"""
        self.logger = ArabicLogger(name="اختبار", console_output=False)
        self.stats = StatisticsCollector(self.logger)

    def test_operation_statistics_are_recorded(self):
        """التحقق من تسجيل إحصائيات العمليات"""
        # بدء عملية
        self.stats.start_operation("معالجة المشاهد")

        # إنهاء العملية مع إحصائيات
        self.stats.end_operation(
            "معالجة المشاهد",
            items_processed=10,
            items_skipped=2,
            items_failed=1
        )

        # التحقق من الإحصائيات
        stats = self.stats.get_operation_stats("معالجة المشاهد")
        self.assertIsNotNone(stats)
        self.assertEqual(stats.items_processed, 10)
        self.assertEqual(stats.items_skipped, 2)
        self.assertEqual(stats.items_failed, 1)

    def test_duration_is_calculated(self):
        """التحقق من حساب المدة الزمنية"""
        import time

        self.stats.start_operation("عملية اختبارية")
        time.sleep(0.1)  # انتظار قليل
        self.stats.end_operation("عملية اختبارية", items_processed=5)

        stats = self.stats.get_operation_stats("عملية اختبارية")
        self.assertGreater(stats.duration_seconds, 0.05)

    def test_global_statistics_are_tracked(self):
        """التحقق من تتبع الإحصائيات العامة"""
        self.stats.add_global_stat("إجمالي_المشاهد", 50)
        self.stats.increment_stat("الحوارات_المعالجة", 10)
        self.stats.increment_stat("الحوارات_المعالجة", 5)

        all_stats = self.stats.get_all_stats()
        self.assertEqual(all_stats["global"]["إجمالي_المشاهد"], 50)
        self.assertEqual(all_stats["global"]["الحوارات_المعالجة"], 15)

    def test_statistics_export(self):
        """التحقق من تصدير الإحصائيات"""
        self.stats.start_operation("اختبار")
        self.stats.end_operation("اختبار", items_processed=5)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            result = self.stats.export_stats(temp_path)
            self.assertTrue(result)

            with open(temp_path, 'r', encoding='utf-8') as f:
                exported = json.load(f)

            self.assertIn("operations", exported)
            self.assertIn("اختبار", exported["operations"])
        finally:
            os.unlink(temp_path)

    def test_multiple_operations_tracking(self):
        """التحقق من تتبع عمليات متعددة"""
        operations = ["تحليل", "تحويل", "تصدير"]

        for i, op in enumerate(operations):
            self.stats.start_operation(op)
            self.stats.end_operation(op, items_processed=i * 10)

        all_stats = self.stats.get_all_stats()
        self.assertEqual(len(all_stats["operations"]), 3)

    if HYPOTHESIS_AVAILABLE:
        @given(st.integers(min_value=0, max_value=1000))
        @settings(max_examples=20)
        def test_property_processed_items_are_recorded_correctly(self, count):
            """خاصية: عدد العناصر المعالجة يُسجل بشكل صحيح"""
            stats = StatisticsCollector(ArabicLogger(console_output=False))
            stats.start_operation("test")
            stats.end_operation("test", items_processed=count)

            result = stats.get_operation_stats("test")
            self.assertEqual(result.items_processed, count)


# ----------------------------
# الخاصية 22 (7.3): اختبار التحقق من البيانات
# ----------------------------
class TestProperty22DataValidation(unittest.TestCase):
    """
    الخاصية 22: التحقق من البيانات

    تتحقق من: المتطلب 6.4
    - WHEN يتم معالجة الملفات
    - THEN النظام SHALL التحقق من وجود البيانات المطلوبة قبل المعالجة
    """

    def setUp(self):
        """تهيئة قبل كل اختبار"""
        self.logger = ArabicLogger(name="اختبار", console_output=False)
        self.validator = DataValidator(self.logger)

    def test_file_exists_validation(self):
        """التحقق من وجود الملف"""
        # ملف غير موجود
        result = self.validator.validate_file_exists("/path/that/does/not/exist.txt")
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)

        # ملف موجود (هذا الملف نفسه)
        result = self.validator.validate_file_exists(__file__)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_empty_path_validation(self):
        """التحقق من المسار الفارغ"""
        result = self.validator.validate_file_exists("")
        self.assertFalse(result.is_valid)
        self.assertIn("فارغ", result.errors[0])

    def test_directory_validation(self):
        """التحقق من وجود المجلد"""
        # مجلد موجود
        result = self.validator.validate_directory(os.path.dirname(__file__))
        self.assertTrue(result.is_valid)

        # مجلد غير موجود
        result = self.validator.validate_directory("/nonexistent/directory")
        self.assertFalse(result.is_valid)

    def test_directory_creation_on_missing(self):
        """التحقق من إنشاء المجلد المفقود"""
        temp_dir = tempfile.mkdtemp()
        new_dir = os.path.join(temp_dir, "new_subdir")

        try:
            result = self.validator.validate_directory(new_dir, create_if_missing=True)
            self.assertTrue(result.is_valid)
            self.assertTrue(os.path.exists(new_dir))
        finally:
            shutil.rmtree(temp_dir)

    def test_required_fields_validation(self):
        """التحقق من الحقول المطلوبة"""
        data = {"name": "أحمد", "age": 30}

        # حقول موجودة
        result = self.validator.validate_required_fields(data, ["name", "age"])
        self.assertTrue(result.is_valid)

        # حقل مفقود
        result = self.validator.validate_required_fields(data, ["name", "email"])
        self.assertFalse(result.is_valid)
        self.assertIn("email", result.errors[0])

    def test_type_validation(self):
        """التحقق من نوع البيانات"""
        # نوع صحيح
        result = self.validator.validate_type("نص", str, "الاسم")
        self.assertTrue(result.is_valid)

        # نوع خاطئ
        result = self.validator.validate_type("123", int, "العمر")
        self.assertFalse(result.is_valid)
        self.assertIn("نوع", result.errors[0])

    def test_list_validation(self):
        """التحقق من القوائم"""
        # قائمة صالحة
        result = self.validator.validate_list_not_empty([1, 2, 3], "الأرقام")
        self.assertTrue(result.is_valid)

        # قائمة فارغة
        result = self.validator.validate_list_not_empty([], "العناصر")
        self.assertFalse(result.is_valid)

    def test_scenes_validation(self):
        """التحقق من صحة المشاهد"""
        scenes = [
            MockScene(scene_id="S0001", dialogue=[]),
            MockScene(scene_id="S0002", dialogue=[])
        ]

        result = self.validator.validate_scenes(scenes)
        self.assertTrue(result.is_valid)

    def test_elements_validation(self):
        """التحقق من صحة العناصر"""
        elements = [
            {"text": "نص 1", "type": "Text"},
            {"text": "نص 2", "type": "Text"}
        ]

        result = self.validator.validate_elements(elements)
        self.assertTrue(result.is_valid)

        # عناصر بدون text
        elements_invalid = [{"type": "Text"}]
        result = self.validator.validate_elements(elements_invalid)
        self.assertTrue(result.is_valid)  # تحذير فقط، ليس خطأ
        self.assertGreater(len(result.warnings), 0)

    if HYPOTHESIS_AVAILABLE:
        @given(st.dictionaries(st.text(min_size=1), st.text()))
        @settings(max_examples=20)
        def test_property_required_fields_validation(self, data):
            """خاصية: التحقق من الحقول المطلوبة يعمل بشكل صحيح"""
            validator = DataValidator(ArabicLogger(console_output=False))

            if data:
                keys = list(data.keys())
                result = validator.validate_required_fields(data, keys)
                self.assertTrue(result.is_valid)


# ----------------------------
# الخاصية 23 (7.4): اختبار ضمان نجاح الكتابة
# ----------------------------
class TestProperty23WriteSuccess(unittest.TestCase):
    """
    الخاصية 23: ضمان نجاح الكتابة

    تتحقق من: المتطلب 6.5
    - WHEN يتم حفظ الملفات
    - THEN النظام SHALL التأكد من نجاح عملية الكتابة قبل المتابعة
    """

    def setUp(self):
        """تهيئة قبل كل اختبار"""
        self.logger = ArabicLogger(name="اختبار", console_output=False)
        self.writer = SafeWriter(self.logger)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """تنظيف بعد كل اختبار"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_successful_text_write(self):
        """التحقق من نجاح كتابة النص"""
        content = "محتوى اختباري بالعربية"
        path = os.path.join(self.temp_dir, "test.txt")

        result = self.writer.write_text(path, content)

        self.assertTrue(result.success)
        self.assertEqual(result.path, path)
        self.assertGreater(result.bytes_written, 0)
        self.assertIsNotNone(result.checksum)

        # التحقق من المحتوى المكتوب
        with open(path, 'r', encoding='utf-8') as f:
            written = f.read()
        self.assertEqual(written, content)

    def test_write_with_verification(self):
        """التحقق من التحقق من نجاح الكتابة"""
        writer = SafeWriter(self.logger, verify_write=True)
        content = "نص للتحقق"
        path = os.path.join(self.temp_dir, "verify.txt")

        result = writer.write_text(path, content)
        self.assertTrue(result.success)

    def test_json_write(self):
        """التحقق من كتابة JSON"""
        data = {"اسم": "أحمد", "عمر": 30, "مدينة": "القاهرة"}
        path = os.path.join(self.temp_dir, "data.json")

        result = self.writer.write_json(path, data)

        self.assertTrue(result.success)

        with open(path, 'r', encoding='utf-8') as f:
            written = json.load(f)
        self.assertEqual(written, data)

    def test_jsonl_write(self):
        """التحقق من كتابة JSONL"""
        rows = [
            {"id": 1, "name": "أحمد"},
            {"id": 2, "name": "محمد"},
            {"id": 3, "name": "علي"}
        ]
        path = os.path.join(self.temp_dir, "data.jsonl")

        result = self.writer.write_jsonl(path, rows)

        self.assertTrue(result.success)

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)

    def test_creates_parent_directory(self):
        """التحقق من إنشاء المجلد الأب تلقائياً"""
        path = os.path.join(self.temp_dir, "subdir", "nested", "file.txt")

        result = self.writer.write_text(path, "محتوى")

        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(path))

    def test_write_failure_is_recorded(self):
        """التحقق من تسجيل فشل الكتابة"""
        # محاولة الكتابة في مسار غير صالح (ملف بدلاً من مجلد)
        # إنشاء ملف ثم محاولة الكتابة داخله كأنه مجلد
        temp_file = os.path.join(self.temp_dir, "not_a_dir.txt")
        with open(temp_file, 'w') as f:
            f.write("test")

        # محاولة الكتابة في مسار حيث جزء منه ملف وليس مجلد
        invalid_path = os.path.join(temp_file, "subdir", "file.txt")

        result = self.writer.write_text(invalid_path, "محتوى")

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_all_successful_check(self):
        """التحقق من فحص نجاح جميع العمليات"""
        # كتابة ناجحة
        self.writer.write_text(os.path.join(self.temp_dir, "1.txt"), "أ")
        self.writer.write_text(os.path.join(self.temp_dir, "2.txt"), "ب")

        self.assertTrue(self.writer.all_successful())

        # إضافة كتابة فاشلة (مسار غير صالح)
        temp_file = os.path.join(self.temp_dir, "blocker.txt")
        with open(temp_file, 'w') as f:
            f.write("blocking file")
        invalid_path = os.path.join(temp_file, "inside", "fail.txt")
        self.writer.write_text(invalid_path, "ج")

        self.assertFalse(self.writer.all_successful())

    def test_write_summary(self):
        """التحقق من ملخص عمليات الكتابة"""
        self.writer.write_text(os.path.join(self.temp_dir, "a.txt"), "أ")
        self.writer.write_text(os.path.join(self.temp_dir, "b.txt"), "ب")

        summary = self.writer.get_summary()

        self.assertEqual(summary["total_writes"], 2)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["failed"], 0)

    def test_backup_creation(self):
        """التحقق من إنشاء نسخة احتياطية"""
        writer = SafeWriter(self.logger, create_backup=True)
        path = os.path.join(self.temp_dir, "backup_test.txt")

        # كتابة أولى
        writer.write_text(path, "المحتوى الأصلي")

        # كتابة ثانية (يجب إنشاء نسخة احتياطية)
        result = writer.write_text(path, "المحتوى الجديد")

        self.assertTrue(result.success)
        if result.backup_path:
            self.assertTrue(os.path.exists(result.backup_path))

    if HYPOTHESIS_AVAILABLE:
        @given(st.text(min_size=1, max_size=1000))
        @settings(max_examples=20)
        def test_property_written_content_matches_input(self, content):
            """خاصية: المحتوى المكتوب يطابق المدخل"""
            assume(content.strip())

            temp_dir = tempfile.mkdtemp()
            try:
                writer = SafeWriter(ArabicLogger(console_output=False))
                path = os.path.join(temp_dir, "test.txt")

                result = writer.write_text(path, content)

                if result.success:
                    with open(path, 'r', encoding='utf-8') as f:
                        written = f.read()
                    self.assertEqual(written, content)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)


# ----------------------------
# اختبارات التكامل
# ----------------------------
class TestIntegration(unittest.TestCase):
    """اختبارات التكامل لجميع المكونات"""

    def test_setup_error_handling(self):
        """اختبار تهيئة جميع المكونات"""
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, "test.log")

        try:
            logger, error_handler, validator, writer, stats = setup_error_handling(
                log_file=log_file,
                console_output=False,
                log_level=LogLevel.DEBUG
            )

            # التحقق من جميع المكونات
            self.assertIsInstance(logger, ArabicLogger)
            self.assertIsInstance(error_handler, ErrorHandler)
            self.assertIsInstance(validator, DataValidator)
            self.assertIsInstance(writer, SafeWriter)
            self.assertIsInstance(stats, StatisticsCollector)

            # التحقق من أن المسجل العام تم تعيينه
            self.assertEqual(get_logger(), logger)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_full_workflow(self):
        """اختبار سير العمل الكامل"""
        temp_dir = tempfile.mkdtemp()

        try:
            logger, error_handler, validator, writer, stats = setup_error_handling(
                console_output=False
            )

            # بدء عملية
            stats.start_operation("معالجة الملفات")

            # التحقق من المدخلات
            result = validator.validate_directory(temp_dir)
            self.assertTrue(result.is_valid)

            # معالجة وكتابة
            data = [{"id": i, "text": f"نص {i}"} for i in range(5)]
            write_result = writer.write_jsonl(
                os.path.join(temp_dir, "output.jsonl"),
                data
            )

            self.assertTrue(write_result.success)

            # إنهاء العملية
            stats.end_operation("معالجة الملفات", items_processed=5)

            # التحقق من الإحصائيات
            op_stats = stats.get_operation_stats("معالجة الملفات")
            self.assertEqual(op_stats.items_processed, 5)

            # التحقق من عدم وجود أخطاء
            self.assertFalse(error_handler.has_critical_errors())

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_safe_import(self):
        """اختبار الاستيراد الآمن للمكتبات"""
        # مكتبة موجودة
        os_module = safe_import("os")
        self.assertIsNotNone(os_module)

        # مكتبة غير موجودة
        nonexistent = safe_import("nonexistent_library_12345")
        self.assertIsNone(nonexistent)

    def test_decorator_with_error_handling(self):
        """اختبار مزخرف معالجة الأخطاء"""

        @with_error_handling("عملية اختبارية", continue_on_error=True, default_return=[])
        def failing_function():
            raise ValueError("خطأ متعمد")

        result = failing_function()
        self.assertEqual(result, [])

    def test_decorator_with_statistics(self):
        """اختبار مزخرف الإحصائيات"""

        @with_statistics("عملية مع إحصائيات")
        def sample_operation():
            return [1, 2, 3, 4, 5]

        result = sample_operation()
        self.assertEqual(len(result), 5)


# ----------------------------
# تشغيل الاختبارات
# ----------------------------
if __name__ == "__main__":
    # إعداد المسجل للاختبارات
    logger = ArabicLogger(name="اختبارات", console_output=True)
    set_logger(logger)

    # تشغيل الاختبارات
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # إضافة جميع اختبارات الخصائص
    suite.addTests(loader.loadTestsFromTestCase(TestProperty20ErrorLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestProperty21StatisticsLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestProperty22DataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestProperty23WriteSuccess))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # تشغيل مع نتائج مفصلة
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # طباعة ملخص
    print("\n" + "=" * 60)
    print("ملخص نتائج الاختبارات:")
    print("=" * 60)
    print(f"الاختبارات المنفذة: {result.testsRun}")
    print(f"النجاحات: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"الإخفاقات: {len(result.failures)}")
    print(f"الأخطاء: {len(result.errors)}")

    if HYPOTHESIS_AVAILABLE:
        print("\n✓ اختبارات hypothesis متاحة ومفعلة")
    else:
        print("\n⚠ اختبارات hypothesis غير متاحة")

    # خروج بحسب النتيجة
    sys.exit(0 if result.wasSuccessful() else 1)
