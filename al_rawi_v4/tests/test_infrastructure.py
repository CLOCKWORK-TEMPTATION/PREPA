# -*- coding: utf-8 -*-
"""
اختبارات وحدة البنية التحتية - نظام الراوي v4.0
================================================

تتضمن:
    - اختبار خاصية 18: معالجة فشل المكتبات (Property-Based Test)
    - اختبارات وحدة للاستيراد الآمن
    - اختبارات حساب التشابه
    - اختبارات نظام التسجيل

المتطلبات المحققة: 5.2, 6.3
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Tuple, Optional


# استيراد وحدة البنية التحتية
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from al_rawi_v4.infrastructure import (
    RAPIDFUZZ_AVAILABLE,
    HYPOTHESIS_AVAILABLE,
    DIFFLIB_AVAILABLE,
    calculate_similarity,
    calculate_similarity_batch,
    get_library_status,
    LibraryStatus,
    LibraryInfo,
    AlRawiLogger,
    setup_logging,
    get_logger,
    AlRawiError,
    LibraryNotAvailableError,
    safe_operation,
    require_library,
    count_arabic_words,
    normalize_arabic_text,
)


# ============================================
# الجزء 1: اختبارات خاصية 18 (Property-Based Tests)
# ============================================

class TestProperty18LibraryFailureHandling(unittest.TestCase):
    """
    **Feature: al-rawi-v4, Property 18: معالجة فشل المكتبات**

    *لأي* مكتبة خارجية تفشل في الاستيراد،
    يجب أن يتعامل النظام مع الاستيراد بطريقة آمنة
    ويتابع العمل بدون التحسين المعتمد عليها

    **تتحقق من: المتطلبات 5.2, 6.3**
    """

    def test_difflib_always_available(self):
        """
        **Property 18.1**: difflib دائماً متوفرة كمكتبة قياسية
        """
        self.assertTrue(DIFFLIB_AVAILABLE)

    def test_library_status_returns_valid_info(self):
        """
        **Property 18.2**: get_library_status تُرجع معلومات صحيحة لجميع المكتبات
        """
        status = get_library_status()

        # يجب أن تحتوي على المكتبات الثلاث
        self.assertIn("rapidfuzz", status)
        self.assertIn("hypothesis", status)
        self.assertIn("difflib", status)

        # كل مكتبة يجب أن تكون من نوع LibraryInfo
        for lib_name, lib_info in status.items():
            self.assertIsInstance(lib_info, LibraryInfo)
            self.assertEqual(lib_info.name, lib_name)
            self.assertIsInstance(lib_info.status, LibraryStatus)

    def test_similarity_works_regardless_of_rapidfuzz(self):
        """
        **Property 18.3**: حساب التشابه يعمل سواء كانت rapidfuzz متوفرة أم لا
        """
        # اختبار مع نصوص متطابقة
        self.assertEqual(calculate_similarity("مرحبا", "مرحبا"), 1.0)

        # اختبار مع نصوص مختلفة تماماً
        self.assertLess(calculate_similarity("مرحبا", "وداعاً"), 0.5)

        # اختبار مع نصوص متشابهة
        similarity = calculate_similarity("محمد أحمد", "محمد احمد")
        self.assertGreater(similarity, 0.8)

    def test_similarity_handles_empty_strings(self):
        """
        **Property 18.4**: حساب التشابه يتعامل مع النصوص الفارغة بأمان
        """
        self.assertEqual(calculate_similarity("", ""), 0.0)
        self.assertEqual(calculate_similarity("نص", ""), 0.0)
        self.assertEqual(calculate_similarity("", "نص"), 0.0)
        self.assertEqual(calculate_similarity(None, "نص"), 0.0)
        self.assertEqual(calculate_similarity("نص", None), 0.0)

    def test_similarity_batch_works_regardless_of_rapidfuzz(self):
        """
        **Property 18.5**: حساب التشابه بالدفعات يعمل بغض النظر عن توفر rapidfuzz
        """
        query = "أحمد"
        choices = ["أحمد علي", "محمد", "أحمد محمود", "فاطمة"]

        results = calculate_similarity_batch(query, choices, threshold=0.5)

        # يجب أن تعود قائمة
        self.assertIsInstance(results, list)

        # يجب أن تحتوي على أزواج (نص، نسبة)
        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], float)

    def test_fallback_info_provided_for_unavailable_libraries(self):
        """
        **Property 18.6**: المكتبات غير المتوفرة توفر معلومات عن البديل
        """
        status = get_library_status()

        rapidfuzz_info = status["rapidfuzz"]

        # rapidfuzz يجب أن يكون لها بديل (difflib)
        self.assertTrue(rapidfuzz_info.fallback_available)
        self.assertEqual(rapidfuzz_info.fallback_name, "difflib")

    def test_error_messages_in_arabic(self):
        """
        **Property 18.7**: رسائل الأخطاء واضحة باللغة العربية
        """
        # اختبار LibraryNotAvailableError
        error = LibraryNotAvailableError("test_lib")
        error_str = str(error)
        self.assertIn("غير متوفرة", error_str)

        # اختبار مع استخدام البديل
        error_with_fallback = LibraryNotAvailableError("test_lib", fallback_used=True)
        error_str = str(error_with_fallback)
        self.assertIn("البديل", error_str)

    def test_safe_operation_decorator_catches_exceptions(self):
        """
        **Property 18.8**: الـ decorator safe_operation يلتقط الاستثناءات ويتابع
        """
        @safe_operation(fallback_value="قيمة افتراضية", log_errors=False)
        def risky_function():
            raise ValueError("خطأ اختباري")

        # يجب أن تعود القيمة الافتراضية بدلاً من رمي استثناء
        result = risky_function()
        self.assertEqual(result, "قيمة افتراضية")

    def test_safe_operation_preserves_successful_results(self):
        """
        **Property 18.9**: الـ decorator safe_operation يحافظ على النتائج الناجحة
        """
        @safe_operation(fallback_value=None, log_errors=False)
        def successful_function(x, y):
            return x + y

        result = successful_function(5, 3)
        self.assertEqual(result, 8)


# ============================================
# الجزء 2: اختبارات نظام التسجيل
# ============================================

class TestLoggingSystem(unittest.TestCase):
    """اختبارات نظام التسجيل المحسّن"""

    def test_logger_creation(self):
        """اختبار إنشاء المسجل"""
        logger = AlRawiLogger(name="test_logger", console_output=False)
        self.assertIsInstance(logger, AlRawiLogger)
        self.assertEqual(logger.name, "test_logger")

    def test_log_levels(self):
        """اختبار مستويات التسجيل"""
        logger = AlRawiLogger(name="test_levels", console_output=False)

        logger.debug("رسالة تصحيح")
        logger.info("رسالة معلومات")
        logger.warning("رسالة تحذير")
        logger.error("رسالة خطأ")
        logger.critical("رسالة حرجة")

        stats = logger.get_stats()
        self.assertEqual(stats["debug"], 1)
        self.assertEqual(stats["info"], 1)
        self.assertEqual(stats["warning"], 1)
        self.assertEqual(stats["error"], 1)
        self.assertEqual(stats["critical"], 1)

    def test_log_library_import(self):
        """اختبار تسجيل استيراد المكتبات"""
        logger = AlRawiLogger(name="test_import", console_output=False)

        logger.log_library_import("test_lib", True)
        logger.log_library_import("failed_lib", False, "سبب الفشل")

        entries = logger.get_entries()
        self.assertEqual(len(entries), 2)

    def test_log_operation_tracking(self):
        """اختبار تتبع العمليات"""
        logger = AlRawiLogger(name="test_ops", console_output=False)

        logger.log_operation_start("عملية_اختبارية", param1="قيمة1")
        logger.log_operation_end("عملية_اختبارية", success=True, result="نجاح")

        entries = logger.get_entries()
        self.assertEqual(len(entries), 2)

    def test_log_statistics(self):
        """اختبار تسجيل الإحصائيات"""
        logger = AlRawiLogger(name="test_stats", console_output=False)

        stats = {"عدد_المشاهد": 10, "عدد_الحوارات": 100}
        logger.log_statistics(stats)

        entries = logger.get_entries()
        self.assertGreater(len(entries), 0)


# ============================================
# الجزء 3: اختبارات الدوال المساعدة
# ============================================

class TestHelperFunctions(unittest.TestCase):
    """اختبارات الدوال المساعدة"""

    def test_count_arabic_words_basic(self):
        """اختبار عدّ الكلمات العربية الأساسي"""
        self.assertEqual(count_arabic_words("مرحبا بالعالم"), 2)
        self.assertEqual(count_arabic_words("كلمة واحدة اثنتان ثلاثة"), 4)
        self.assertEqual(count_arabic_words(""), 0)
        self.assertEqual(count_arabic_words(None), 0)

    def test_count_arabic_words_with_extra_spaces(self):
        """اختبار عدّ الكلمات مع مسافات زائدة"""
        self.assertEqual(count_arabic_words("  مرحبا   بالعالم  "), 2)
        self.assertEqual(count_arabic_words("كلمة\n\nأخرى"), 2)

    def test_normalize_arabic_text_removes_diacritics(self):
        """اختبار إزالة التشكيل"""
        normalized = normalize_arabic_text("مُحَمَّد")
        self.assertNotIn("ُ", normalized)
        self.assertNotIn("َ", normalized)
        self.assertNotIn("ّ", normalized)

    def test_normalize_arabic_text_unifies_letters(self):
        """اختبار توحيد الحروف"""
        # توحيد الهمزات
        self.assertEqual(normalize_arabic_text("إسماعيل"), normalize_arabic_text("اسماعيل"))
        self.assertEqual(normalize_arabic_text("أحمد"), normalize_arabic_text("احمد"))

        # توحيد التاء المربوطة
        self.assertEqual(normalize_arabic_text("فاطمة"), normalize_arabic_text("فاطمه"))

    def test_normalize_arabic_text_handles_empty(self):
        """اختبار التعامل مع النصوص الفارغة"""
        self.assertEqual(normalize_arabic_text(""), "")
        self.assertEqual(normalize_arabic_text(None), "")


# ============================================
# الجزء 4: اختبارات التشابه المتقدمة
# ============================================

class TestSimilarityCalculations(unittest.TestCase):
    """اختبارات حساب التشابه المتقدمة"""

    def test_identical_strings_return_one(self):
        """النصوص المتطابقة تعيد 1.0"""
        test_cases = [
            "مرحبا",
            "أحمد محمد علي",
            "السلام عليكم ورحمة الله",
            "1234",
            "Hello مرحبا",
        ]

        for text in test_cases:
            with self.subTest(text=text):
                self.assertEqual(calculate_similarity(text, text), 1.0)

    def test_completely_different_strings_return_low(self):
        """النصوص المختلفة تماماً تعيد قيمة منخفضة"""
        similarity = calculate_similarity("أحمد", "سيارة")
        self.assertLess(similarity, 0.3)

    def test_similar_arabic_names(self):
        """الأسماء العربية المتشابهة"""
        # اسم مختصر واسم كامل
        similarity = calculate_similarity("محمد", "محمد أحمد")
        self.assertGreater(similarity, 0.5)

        # اختلافات إملائية بسيطة
        similarity = calculate_similarity("أحمد", "احمد")
        self.assertGreater(similarity, 0.8)

    def test_batch_similarity_returns_sorted_results(self):
        """نتائج الدفعات مرتبة تنازلياً"""
        query = "محمد"
        choices = ["محمد أحمد", "أحمد محمد", "محمود", "علي"]

        results = calculate_similarity_batch(query, choices, threshold=0.3)

        # التحقق من الترتيب التنازلي
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ============================================
# الجزء 5: اختبارات خاصية hypothesis (إذا كانت متوفرة)
# ============================================

try:
    from hypothesis import given, strategies as st, settings

    class TestPropertyBasedWithHypothesis(unittest.TestCase):
        """
        اختبارات خاصية باستخدام hypothesis

        **Feature: al-rawi-v4, Property 18: معالجة فشل المكتبات**
        """

        @settings(max_examples=100)
        @given(st.text(min_size=1, max_size=50))
        def test_similarity_self_is_one(self, text):
            """
            **Property 18.10**: أي نص متطابق مع نفسه يعيد 1.0
            """
            if text.strip():  # تجاهل النصوص الفارغة
                self.assertEqual(calculate_similarity(text, text), 1.0)

        @settings(max_examples=100)
        @given(st.text(min_size=1, max_size=30), st.text(min_size=1, max_size=30))
        def test_similarity_is_symmetric(self, s1, s2):
            """
            **Property 18.11**: التشابه متماثل: sim(a,b) == sim(b,a)
            """
            if s1.strip() and s2.strip():
                sim1 = calculate_similarity(s1, s2)
                sim2 = calculate_similarity(s2, s1)
                self.assertAlmostEqual(sim1, sim2, places=5)

        @settings(max_examples=100)
        @given(st.text(min_size=1, max_size=30), st.text(min_size=1, max_size=30))
        def test_similarity_range_is_valid(self, s1, s2):
            """
            **Property 18.12**: نسبة التشابه دائماً بين 0 و 1
            """
            similarity = calculate_similarity(s1, s2)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)

        @settings(max_examples=100)
        @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
        def test_batch_similarity_never_crashes(self, choices):
            """
            **Property 18.13**: حساب التشابه بالدفعات لا يسبب أخطاء أبداً
            """
            query = "اختبار"
            try:
                results = calculate_similarity_batch(query, choices, threshold=0.0)
                self.assertIsInstance(results, list)
            except Exception as e:
                self.fail(f"calculate_similarity_batch raised {e}")

except ImportError:
    # hypothesis غير متوفرة، تخطي هذه الاختبارات
    pass


# ============================================
# تشغيل الاختبارات
# ============================================

if __name__ == "__main__":
    # طباعة حالة المكتبات قبل الاختبارات
    print("\n" + "=" * 60)
    print("اختبارات البنية التحتية - نظام الراوي v4.0")
    print("=" * 60)
    print(f"\nحالة المكتبات:")
    print(f"  - rapidfuzz: {'متوفرة' if RAPIDFUZZ_AVAILABLE else 'غير متوفرة'}")
    print(f"  - hypothesis: {'متوفرة' if HYPOTHESIS_AVAILABLE else 'غير متوفرة'}")
    print(f"  - difflib: متوفرة (مكتبة قياسية)")
    print("\n" + "=" * 60 + "\n")

    unittest.main(verbosity=2)
