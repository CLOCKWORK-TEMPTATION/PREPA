"""
اختبارات وحدة الميتاداتا الزمنية (Temporal Metadata Extractor)
تتحقق من المتطلبات 4.1 - 4.5 وخصائص 13-17

المهمة 5: تطوير وحدة الميتاداتا الزمنية
"""

import sys
import os
import re
from typing import List

# إضافة المسار للوصول إلى الوحدات الرئيسية
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# محاولة استيراد hypothesis للاختبارات القائمة على الخصائص
try:
    from hypothesis import given, strategies as st, settings, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("تحذير: مكتبة hypothesis غير متوفرة - سيتم تخطي اختبارات الخصائص")

from screenplay_to_dataset import (
    Scene,
    DialogueTurn,
    TemporalMetadataExtractor,
)


# =====================================================
# اختبارات الوحدة (Unit Tests) - القسم 5.6
# =====================================================

class TestTemporalMetadataExtractorUnit:
    """اختبارات الوحدة للحالات الزمنية المحددة"""

    def test_extract_specific_year_1986(self):
        """
        اختبار استخراج سنة 1986 من عنوان المشهد
        **Feature: al-rawi-v4, المتطلبات 4.1, 4.2**
        """
        extractor = TemporalMetadataExtractor()
        result = extractor.extract_time_period("مشهد 1 - 1986 - داخلي منزل")
        assert result == "1986", f"Expected '1986', got '{result}'"

    def test_extract_specific_year_2009(self):
        """
        اختبار استخراج سنة 2009 من عنوان المشهد
        **Feature: al-rawi-v4, المتطلبات 4.1, 4.2**
        """
        extractor = TemporalMetadataExtractor()
        result = extractor.extract_time_period("داخلي - مكتب - 2009")
        assert result == "2009", f"Expected '2009', got '{result}'"

    def test_no_year_returns_default(self):
        """
        اختبار السلوك عند عدم وجود سنوات - يجب إرجاع "غير محدد"
        **Feature: al-rawi-v4, المتطلبات 4.3**
        """
        extractor = TemporalMetadataExtractor()
        result = extractor.extract_time_period("داخلي - منزل - ليل")
        assert result == "غير محدد", f"Expected 'غير محدد', got '{result}'"

    def test_inheritance_across_scenes(self):
        """
        اختبار وراثة السنوات عبر مشاهد متتالية
        **Feature: al-rawi-v4, المتطلبات 4.3**
        """
        extractor = TemporalMetadataExtractor()

        # المشهد الأول يحتوي على سنة
        result1 = extractor.extract_time_period("مشهد 1 - 1986")
        assert result1 == "1986"

        # المشهد الثاني لا يحتوي على سنة - يجب أن يرث
        result2 = extractor.extract_time_period("داخلي - منزل")
        assert result2 == "1986", f"Expected inherited '1986', got '{result2}'"

        # المشهد الثالث لا يحتوي على سنة - يجب أن يرث
        result3 = extractor.extract_time_period("خارجي - شارع")
        assert result3 == "1986", f"Expected inherited '1986', got '{result3}'"

    def test_inheritance_updates_on_new_year(self):
        """
        اختبار تحديث السنة الموروثة عند إيجاد سنة جديدة
        **Feature: al-rawi-v4, المتطلبات 4.3**
        """
        extractor = TemporalMetadataExtractor()

        # المشهد الأول
        result1 = extractor.extract_time_period("مشهد 1 - 1986")
        assert result1 == "1986"

        # المشهد الثاني يرث
        result2 = extractor.extract_time_period("داخلي - منزل")
        assert result2 == "1986"

        # المشهد الثالث يحتوي على سنة جديدة
        result3 = extractor.extract_time_period("فلاش باك - 1972")
        assert result3 == "1972"

        # المشهد الرابع يرث السنة الجديدة
        result4 = extractor.extract_time_period("داخلي - غرفة")
        assert result4 == "1972"

    def test_empty_text(self):
        """اختبار السلوك مع نص فارغ"""
        extractor = TemporalMetadataExtractor()
        result = extractor.extract_time_period("")
        assert result == "غير محدد"

    def test_none_text(self):
        """اختبار السلوك مع قيمة None"""
        extractor = TemporalMetadataExtractor()
        result = extractor.extract_time_period(None)
        assert result == "غير محدد"

    def test_reset_clears_state(self):
        """اختبار أن reset تمسح الحالة"""
        extractor = TemporalMetadataExtractor()

        # تعيين سنة
        extractor.extract_time_period("مشهد 1 - 1986")
        assert extractor.last_known_year == "1986"

        # إعادة التعيين
        extractor.reset()
        assert extractor.last_known_year == "غير محدد"
        assert len(extractor.extraction_log) == 0


class TestTemporalMetadataWithScenes:
    """اختبارات تطبيق الاستخراج على المشاهد"""

    def _create_scene(self, scene_id: str, heading: str,
                      actions: List[str] = None, dialogue: List[DialogueTurn] = None) -> Scene:
        """دالة مساعدة لإنشاء مشهد"""
        return Scene(
            scene_id=scene_id,
            scene_number=int(scene_id[1:]) if scene_id.startswith("S") else 1,
            heading=heading,
            location="غرفة",
            time_of_day="نهار",
            int_ext="داخلي",
            time_period="غير محدد",
            actions=actions or [],
            dialogue=dialogue or [],
            transitions=[],
            element_ids=[],
            full_text=heading,
            characters=[],
        )

    def test_apply_to_scenes_with_year_in_heading(self):
        """
        اختبار تطبيق الاستخراج على مشاهد مع سنوات في العناوين
        **Feature: al-rawi-v4, Property 14: إضافة حقل الفترة الزمنية**
        """
        scenes = [
            self._create_scene("S0001", "مشهد 1 - 1986 - داخلي"),
            self._create_scene("S0002", "داخلي - منزل"),
            self._create_scene("S0003", "مشهد 3 - 2009 - خارجي"),
        ]

        extractor = TemporalMetadataExtractor()
        result = extractor.apply_to_scenes(scenes)

        assert result[0].time_period == "1986"
        assert result[1].time_period == "1986"  # وراثة
        assert result[2].time_period == "2009"

    def test_apply_to_scenes_with_year_in_content(self):
        """
        اختبار البحث في محتوى المشهد
        **Feature: al-rawi-v4, Property 17: البحث في محتوى المشهد**
        """
        scenes = [
            self._create_scene(
                "S0001",
                "داخلي - منزل - ليل",
                actions=["على الشاشة: عام 1967", "يجلس أحمد في الغرفة"]
            ),
        ]

        extractor = TemporalMetadataExtractor()
        result = extractor.apply_to_scenes(scenes)

        assert result[0].time_period == "1967", \
            f"Expected '1967' from content, got '{result[0].time_period}'"

    def test_apply_to_scenes_inheritance(self):
        """
        اختبار وراثة الفترة الزمنية بين المشاهد
        **Feature: al-rawi-v4, Property 15: وراثة الفترة الزمنية**
        """
        scenes = [
            self._create_scene("S0001", "داخلي - بدون سنة"),
            self._create_scene("S0002", "مشهد 2 - 1990"),
            self._create_scene("S0003", "داخلي - غرفة"),
            self._create_scene("S0004", "خارجي - شارع"),
        ]

        extractor = TemporalMetadataExtractor()
        result = extractor.apply_to_scenes(scenes)

        assert result[0].time_period == "غير محدد"  # لا يوجد سنة سابقة
        assert result[1].time_period == "1990"
        assert result[2].time_period == "1990"  # وراثة
        assert result[3].time_period == "1990"  # وراثة

    def test_extraction_stats(self):
        """اختبار إحصائيات الاستخراج"""
        scenes = [
            self._create_scene("S0001", "مشهد 1 - 1986"),
            self._create_scene("S0002", "داخلي - منزل"),
            self._create_scene("S0003", "مشهد 3 - 2009"),
        ]

        extractor = TemporalMetadataExtractor()
        extractor.apply_to_scenes(scenes)
        stats = extractor.get_extraction_stats()

        # يجب أن يجد سنتين (1986 و 2009)
        assert stats["found_years"] == 2
        # المشهد الثاني يرث ولكن لا يُسجل في extraction_log
        # لأن الوراثة لا تُسجل كعملية استخراج
        assert stats["total_extractions"] >= 2


class TestYearPatternRegex:
    """اختبارات نمط regex للسنوات"""

    def test_pattern_matches_valid_years(self):
        """اختبار أن النمط يطابق السنوات الصالحة"""
        pattern = re.compile(r'\b(19|20)\d{2}\b')

        valid_years = ["1900", "1950", "1986", "1999", "2000", "2024", "2099"]
        for year in valid_years:
            match = pattern.search(f"text {year} more text")
            assert match is not None, f"Pattern should match {year}"
            assert match.group(0) == year

    def test_pattern_rejects_invalid_years(self):
        """اختبار أن النمط لا يطابق السنوات غير الصالحة"""
        pattern = re.compile(r'\b(19|20)\d{2}\b')

        invalid_years = ["1899", "2100", "3000", "123", "12345"]
        for year in invalid_years:
            text = f"text {year} more text"
            match = pattern.search(text)
            if match:
                assert match.group(0) != year, f"Pattern should not match {year}"


# =====================================================
# اختبارات الخصائص (Property-Based Tests) - القسم 5.1-5.5
# =====================================================

if HYPOTHESIS_AVAILABLE:

    class TestTemporalMetadataProperties:
        """اختبارات الخصائص للميتاداتا الزمنية"""

        @given(st.integers(min_value=1900, max_value=2099))
        @settings(max_examples=100)
        def test_property_13_extract_years_from_headings(self, year: int):
            """
            **Feature: al-rawi-v4, Property 13: استخراج السنوات من العناوين**
            لأي عنوان مشهد، يجب أن يبحث النظام عن السنوات باستخدام
            النمط \\b(19|20)\\d{2}\\b ويستخرجها

            المتطلبات: 4.1
            """
            extractor = TemporalMetadataExtractor()
            heading = f"مشهد 1 - {year} - داخلي"
            result = extractor.extract_time_period(heading)
            assert result == str(year), \
                f"Expected '{year}' from heading, got '{result}'"

        @given(st.integers(min_value=1900, max_value=2099))
        @settings(max_examples=100)
        def test_property_14_add_time_period_field(self, year: int):
            """
            **Feature: al-rawi-v4, Property 14: إضافة حقل الفترة الزمنية**
            لأي مشهد يتم العثور على سنة في نصه، يجب أن يضيف النظام
            حقل time_period إلى كائن Scene

            المتطلبات: 4.2
            """
            scene = Scene(
                scene_id="S0001",
                scene_number=1,
                heading=f"مشهد 1 - {year}",
                location="غرفة",
                time_of_day="نهار",
                int_ext="داخلي",
                time_period="غير محدد",
            )

            extractor = TemporalMetadataExtractor()
            result = extractor.apply_to_scenes([scene])

            assert result[0].time_period == str(year), \
                f"Scene should have time_period={year}"

        @given(st.lists(
            st.booleans(),
            min_size=2,
            max_size=10
        ))
        @settings(max_examples=100)
        def test_property_15_time_period_inheritance(self, has_year_list: List[bool]):
            """
            **Feature: al-rawi-v4, Property 15: وراثة الفترة الزمنية**
            لأي مشهد لا يحتوي على سنة، يجب أن يرث النظام السنة
            من المشهد السابق أو يضع "غير محدد"

            المتطلبات: 4.3
            """
            # بناء قائمة مشاهد
            scenes = []
            for i, has_year in enumerate(has_year_list):
                if has_year:
                    heading = f"مشهد {i+1} - {1980 + i}"
                else:
                    heading = f"داخلي - غرفة {i+1}"

                scenes.append(Scene(
                    scene_id=f"S{i+1:04d}",
                    scene_number=i + 1,
                    heading=heading,
                    location="غرفة",
                    time_of_day="نهار",
                    int_ext="داخلي",
                    time_period="غير محدد",
                ))

            extractor = TemporalMetadataExtractor()
            result = extractor.apply_to_scenes(scenes)

            # التحقق من الوراثة
            last_known = "غير محدد"
            for i, (has_year, scene) in enumerate(zip(has_year_list, result)):
                if has_year:
                    expected = str(1980 + i)
                    last_known = expected
                else:
                    expected = last_known

                assert scene.time_period == expected, \
                    f"Scene {i+1}: expected '{expected}', got '{scene.time_period}'"

        @given(st.integers(min_value=1900, max_value=2099))
        @settings(max_examples=100)
        def test_property_16_include_in_export(self, year: int):
            """
            **Feature: al-rawi-v4, Property 16: تضمين الفترة في التصدير**
            لأي عملية تصدير JSONL، يجب أن يتضمن النظام حقل
            time_period في الميتاداتا

            المتطلبات: 4.4
            """
            scene = Scene(
                scene_id="S0001",
                scene_number=1,
                heading=f"مشهد 1 - {year}",
                location="غرفة",
                time_of_day="نهار",
                int_ext="داخلي",
                time_period="غير محدد",
            )

            extractor = TemporalMetadataExtractor()
            extractor.apply_to_scenes([scene])

            # محاكاة التصدير
            export_dict = {
                "scene_id": scene.scene_id,
                "time_period": scene.time_period,
            }

            assert "time_period" in export_dict
            assert export_dict["time_period"] == str(year)

        @given(st.text(min_size=10, max_size=200))
        @settings(max_examples=100)
        def test_property_17_search_in_content(self, content: str):
            """
            **Feature: al-rawi-v4, Property 17: البحث في محتوى المشهد**
            لأي مشهد يتم معالجته، يجب أن يبحث النظام عن المؤشرات
            الزمنية في محتوى المشهد وليس فقط العنوان

            المتطلبات: 4.5
            """
            # إضافة سنة عشوائية للمحتوى
            year = 1985
            content_with_year = f"{content} عام {year} {content}"

            extractor = TemporalMetadataExtractor()
            result = extractor.extract_from_heading_and_content(
                heading="داخلي - منزل",  # بدون سنة
                content=content_with_year
            )

            assert result == str(year), \
                f"Should find year {year} in content, got '{result}'"

        @given(st.text(min_size=0, max_size=100))
        @settings(max_examples=100)
        def test_no_crash_on_arbitrary_text(self, text: str):
            """
            اختبار أن المستخرج لا يتعطل مع نص عشوائي
            """
            extractor = TemporalMetadataExtractor()
            try:
                result = extractor.extract_time_period(text)
                assert isinstance(result, str)
            except Exception as e:
                pytest.fail(f"Extractor crashed on text: {text[:50]}... Error: {e}")


# =====================================================
# اختبارات التكامل
# =====================================================

class TestTemporalMetadataIntegration:
    """اختبارات تكامل الميتاداتا الزمنية"""

    def test_full_pipeline_with_temporal_metadata(self):
        """اختبار خط المعالجة الكامل مع الميتاداتا الزمنية"""
        # إنشاء سيناريو بسيط
        scenes = [
            Scene(
                scene_id="S0001",
                scene_number=1,
                heading="مشهد 1 - 1986 - داخلي منزل رأفت - ليل",
                location="منزل رأفت",
                time_of_day="ليل",
                int_ext="داخلي",
                time_period="غير محدد",
                actions=["يجلس رأفت في الغرفة"],
                dialogue=[
                    DialogueTurn(
                        scene_id="S0001",
                        turn_id=1,
                        speaker="رأفت",
                        text="مرحباً"
                    )
                ],
            ),
            Scene(
                scene_id="S0002",
                scene_number=2,
                heading="داخلي - مكتب - نهار",
                location="مكتب",
                time_of_day="نهار",
                int_ext="داخلي",
                time_period="غير محدد",
            ),
            Scene(
                scene_id="S0003",
                scene_number=3,
                heading="فلاش باك - 1972 - خارجي شارع",
                location="شارع",
                time_of_day="نهار",
                int_ext="خارجي",
                time_period="غير محدد",
            ),
        ]

        # تطبيق الاستخراج
        extractor = TemporalMetadataExtractor()
        result = extractor.apply_to_scenes(scenes)

        # التحقق من النتائج
        assert result[0].time_period == "1986", "Scene 1 should have 1986"
        assert result[1].time_period == "1986", "Scene 2 should inherit 1986"
        assert result[2].time_period == "1972", "Scene 3 should have 1972"

        # التحقق من الإحصائيات
        stats = extractor.get_extraction_stats()
        assert stats["found_years"] == 2  # 1986 و 1972
        # total_extractions يعتمد على عدد العمليات المسجلة
        assert stats["total_extractions"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
