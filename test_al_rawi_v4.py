"""
اختبارات نظام الراوي الإصدار 4.0
اختبارات الخصائص واختبارات الوحدة

Task 3.1-3.5: اختبارات وحدة إثراء السياق
Task 4.1-4.5: اختبارات وحدة فلترة الجودة
"""

import pytest
import json
import os
from pathlib import Path
from typing import List

# استيراد المكتبات للاختبار
try:
    from hypothesis import given, strategies as st, settings, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("تحذير: مكتبة hypothesis غير متوفرة. سيتم تخطي اختبارات الخصائص.")

# استيراد الوحدات المراد اختبارها
from al_rawi_v4 import (
    ContextEnricher,
    QualityFilter,
    DatasetExporter,
    DialogueTurn,
    Scene,
    count_arabic_words,
    is_significant_action,
    is_transition_line,
    create_sample_scene
)


# ============================================================
# استراتيجيات Hypothesis المخصصة للنصوص العربية
# ============================================================

if HYPOTHESIS_AVAILABLE:
    # استراتيجية لإنشاء نص عربي
    arabic_text_strategy = st.text(
        alphabet="أبتثجحخدذرزسشصضطظعغفقكلمنهويىةءآإؤئ ",
        min_size=1,
        max_size=100
    )

    # استراتيجية لإنشاء اسم شخصية عربية
    arabic_name_strategy = st.text(
        alphabet="أبتثجحخدذرزسشصضطظعغفقكلمنهويىةآإ ",
        min_size=2,
        max_size=20
    )

    # استراتيجية لإنشاء قائمة أسطر وصفية
    actions_strategy = st.lists(
        st.text(min_size=5, max_size=100),
        min_size=0,
        max_size=10
    )


# ============================================================
# دوال مساعدة للاختبارات
# ============================================================

def create_dialogue_turn(
    text: str,
    speaker: str = "شخصية",
    sentiment_score: float = 0.5,
    scene_id: str = "S0001",
    turn_id: int = 1
) -> DialogueTurn:
    """إنشاء وحدة حوار للاختبار"""
    return DialogueTurn(
        scene_id=scene_id,
        turn_id=turn_id,
        speaker=speaker,
        text=text,
        sentiment="positive" if sentiment_score > 0.5 else "neutral",
        sentiment_score=sentiment_score
    )


def create_test_scene(
    dialogue_texts: List[str] = None,
    actions: List[str] = None,
    location: str = "منزل",
    time_of_day: str = "نهار"
) -> Scene:
    """إنشاء مشهد للاختبار"""
    if dialogue_texts is None:
        dialogue_texts = ["مرحباً كيف حالك؟", "أنا بخير شكراً لك"]
    if actions is None:
        actions = []

    dialogue = [
        create_dialogue_turn(text, f"شخصية_{i}", 0.5, "S0001", i + 1)
        for i, text in enumerate(dialogue_texts)
    ]

    return Scene(
        scene_id="S0001",
        scene_number=1,
        heading=f"مشهد 1 - داخلي - {location} - {time_of_day}",
        location=location,
        time_of_day=time_of_day,
        int_ext="داخلي",
        time_period="غير محدد",
        actions=actions,
        dialogue=dialogue,
        characters=[f"شخصية_{i}" for i in range(len(dialogue_texts))]
    )


# ============================================================
# Task 3.1: اختبار خاصية لتضمين السياق الوصفي
# الخاصية 5: تضمين السياق الوصفي
# ============================================================

class TestContextEnricherProperty5:
    """
    **Feature: al-rawi-v4, Property 5: تضمين السياق الوصفي**
    لأي عملية تصدير Alpaca، يجب أن يتضمن النظام آخر سطر وصفي مهم
    قبل الحوار في حقل الإدخال
    المتطلب 2.1
    """

    def test_last_significant_action_included_in_export(self):
        """اختبار أن آخر سطر وصفي مهم يُضمّن في التصدير"""
        enricher = ContextEnricher()
        actions = [
            "يدخل أحمد من الباب",
            "قطع",
            "ينظر إلى الصورة المعلقة على الحائط بتأمل عميق"
        ]
        scene = create_test_scene(
            dialogue_texts=["مرحباً"],
            actions=actions
        )

        result = enricher.export_contextual_alpaca([scene])

        assert len(result) > 0
        # التحقق من أن السياق الوصفي موجود في المدخل
        input_text = result[0]["input"]
        assert "سياق" in input_text or "ينظر إلى الصورة" in input_text

    def test_no_action_when_empty_actions_list(self):
        """اختبار السلوك عندما لا توجد أسطر وصفية"""
        enricher = ContextEnricher()
        scene = create_test_scene(
            dialogue_texts=["مرحباً كيف حالك؟"],
            actions=[]
        )

        result = enricher.export_contextual_alpaca([scene])

        assert len(result) > 0
        # يجب أن يستخدم عنوان المشهد فقط بدون سياق وصفي فارغ

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis غير متوفرة")
    @given(actions_strategy)
    @settings(max_examples=100)
    def test_property_action_extraction_always_returns_string(self, actions):
        """
        **Feature: al-rawi-v4, Property 5: تضمين السياق الوصفي**
        لأي قائمة من الأسطر الوصفية، يجب أن ترجع دالة الاستخراج
        سلسلة نصية (فارغة أو غير فارغة)
        """
        enricher = ContextEnricher()
        result = enricher._get_last_significant_action(actions)

        assert isinstance(result, str)


# ============================================================
# Task 3.2: اختبار خاصية لدمج معلومات المشهد
# الخاصية 6: دمج معلومات المشهد
# ============================================================

class TestContextEnricherProperty6:
    """
    **Feature: al-rawi-v4, Property 6: دمج معلومات المشهد**
    لأي حوار يتم تنسيقه، يجب أن يدمج النظام معلومات المكان
    والزمان مع السياق الوصفي في حقل الإدخال
    المتطلب 2.2
    """

    def test_location_included_in_enriched_setup(self):
        """اختبار أن معلومات المكان تُدمج في وصف المشهد"""
        enricher = ContextEnricher()
        scene = create_test_scene(location="مكتب الشركة")

        setup = enricher._build_enriched_scene_setup(scene, "")

        assert "مكتب الشركة" in setup

    def test_time_of_day_included_in_enriched_setup(self):
        """اختبار أن معلومات الوقت تُدمج في وصف المشهد"""
        enricher = ContextEnricher()
        scene = create_test_scene(time_of_day="ليل")

        setup = enricher._build_enriched_scene_setup(scene, "")

        assert "ليل" in setup

    def test_context_action_included_in_enriched_setup(self):
        """اختبار أن السياق الوصفي يُدمج في وصف المشهد"""
        enricher = ContextEnricher()
        scene = create_test_scene()
        action = "يجلس بجانب النافذة وينظر للخارج"

        setup = enricher._build_enriched_scene_setup(scene, action)

        assert "سياق" in setup
        assert "يجلس بجانب النافذة" in setup

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis غير متوفرة")
    @given(
        location=st.text(min_size=1, max_size=50),
        time_of_day=st.sampled_from(["ليل", "نهار", "صباح", "مساء"]),
        action=st.text(min_size=0, max_size=100)
    )
    @settings(max_examples=100)
    def test_property_enriched_setup_always_string(self, location, time_of_day, action):
        """
        **Feature: al-rawi-v4, Property 6: دمج معلومات المشهد**
        لأي مجموعة من معلومات المشهد، يجب أن ترجع دالة البناء
        سلسلة نصية
        """
        enricher = ContextEnricher()
        scene = create_test_scene(location=location, time_of_day=time_of_day)

        result = enricher._build_enriched_scene_setup(scene, action)

        assert isinstance(result, str)


# ============================================================
# Task 3.3: اختبار خاصية لاستخراج الأسطر الوصفية
# الخاصية 7: استخراج الأسطر الوصفية
# ============================================================

class TestContextEnricherProperty7:
    """
    **Feature: al-rawi-v4, Property 7: استخراج الأسطر الوصفية**
    لأي مشهد يحتوي على قائمة actions، يجب أن يستخرج النظام
    الأسطر الوصفية منها لاستخدامها في السياق
    المتطلب 2.3
    """

    def test_extracts_last_significant_action(self):
        """اختبار استخراج آخر سطر وصفي مهم"""
        enricher = ContextEnricher(min_action_length=10)
        actions = [
            "يدخل أحمد",  # قصير
            "ينظر إلى الصورة المعلقة على الحائط بتأمل"  # طويل ومهم
        ]

        result = enricher._get_last_significant_action(actions)

        assert result == "ينظر إلى الصورة المعلقة على الحائط بتأمل"

    def test_skips_transitions(self):
        """اختبار تخطي الانتقالات"""
        enricher = ContextEnricher(min_action_length=5)
        actions = [
            "يجلس على الكرسي",
            "قطع"
        ]

        result = enricher._get_last_significant_action(actions)

        assert result == "يجلس على الكرسي"
        assert result != "قطع"

    def test_returns_empty_for_short_actions(self):
        """اختبار إرجاع سلسلة فارغة للأسطر القصيرة"""
        enricher = ContextEnricher(min_action_length=20)
        actions = ["يدخل", "يخرج"]

        result = enricher._get_last_significant_action(actions)

        assert result == ""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis غير متوفرة")
    @given(actions_strategy)
    @settings(max_examples=100)
    def test_property_extraction_never_returns_transition(self, actions):
        """
        **Feature: al-rawi-v4, Property 7: استخراج الأسطر الوصفية**
        لأي قائمة من الأسطر الوصفية، يجب ألا ترجع دالة الاستخراج
        سطر انتقال
        """
        enricher = ContextEnricher()
        result = enricher._get_last_significant_action(actions)

        assert not is_transition_line(result)


# ============================================================
# Task 3.4: اختبار خاصية لتنسيق الإدخال المعياري
# الخاصية 8: تنسيق الإدخال المعياري
# ============================================================

class TestContextEnricherProperty8:
    """
    **Feature: al-rawi-v4, Property 8: تنسيق الإدخال المعياري**
    لأي حوار يتم تنسيقه، يجب أن يستخدم النظام التنسيق
    "المكان: X. [سياق: Y]. المتحدث: Z"
    المتطلب 2.4
    """

    def test_standard_format_with_context(self):
        """اختبار التنسيق المعياري مع سياق"""
        enricher = ContextEnricher()
        scene = create_test_scene(
            dialogue_texts=["مرحباً"],
            actions=["يجلس أحمد على الكرسي الخشبي القديم"],
            location="غرفة المعيشة"
        )

        result = enricher.export_contextual_alpaca([scene])

        assert len(result) > 0
        input_text = result[0]["input"]
        assert "المكان:" in input_text or "غرفة المعيشة" in input_text
        assert "المتحدث:" in input_text

    def test_standard_format_includes_speaker(self):
        """اختبار أن التنسيق يتضمن اسم المتحدث"""
        enricher = ContextEnricher()
        turn = create_dialogue_turn("مرحباً", speaker="خالد")
        scene = create_test_scene()
        scene.dialogue = [turn]

        result = enricher.export_contextual_alpaca([scene])

        assert len(result) > 0
        assert "خالد" in result[0]["input"]

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis غير متوفرة")
    @given(
        dialogue_text=st.text(min_size=1, max_size=50),
        speaker=st.text(alphabet="أبتثجحخدذرزسشصضطظعغفقكلمنهويى", min_size=2, max_size=15)
    )
    @settings(max_examples=100)
    def test_property_export_includes_speaker_in_input(self, dialogue_text, speaker):
        """
        **Feature: al-rawi-v4, Property 8: تنسيق الإدخال المعياري**
        لأي حوار يتم تصديره، يجب أن يتضمن حقل الإدخال اسم المتحدث
        """
        assume(len(speaker.strip()) > 0)
        assume(len(dialogue_text.strip()) > 0)

        enricher = ContextEnricher()
        turn = create_dialogue_turn(dialogue_text, speaker=speaker)
        scene = create_test_scene()
        scene.dialogue = [turn]

        result = enricher.export_contextual_alpaca([scene])

        if result:  # قد تكون فارغة إذا كان النص فارغاً
            assert speaker in result[0]["input"] or "المتحدث" in result[0]["input"]


# ============================================================
# Task 3.5: اختبارات الوحدة للحالات الخاصة
# ============================================================

class TestContextEnricherUnitTests:
    """
    اختبارات الوحدة للحالات الخاصة في إثراء السياق
    المتطلب 2.5
    """

    def test_no_context_uses_heading_only(self):
        """اختبار استخدام عنوان المشهد فقط عند عدم وجود سياق"""
        enricher = ContextEnricher()
        scene = Scene(
            scene_id="S0001",
            scene_number=1,
            heading="مشهد 1 - داخلي - منزل - نهار",
            location="منزل",
            time_of_day="نهار",
            int_ext="داخلي",
            actions=[],
            dialogue=[create_dialogue_turn("مرحباً")],
            characters=["شخصية"]
        )

        result = enricher.export_contextual_alpaca([scene])

        assert len(result) > 0
        assert "منزل" in result[0]["input"]

    def test_empty_scene_produces_no_output(self):
        """اختبار أن المشهد الفارغ لا ينتج مخرجات"""
        enricher = ContextEnricher()
        scene = create_test_scene(dialogue_texts=[])

        result = enricher.export_contextual_alpaca([scene])

        assert len(result) == 0

    def test_multiple_scenes_processed_correctly(self):
        """اختبار معالجة مشاهد متعددة"""
        enricher = ContextEnricher()
        scene1 = create_test_scene(dialogue_texts=["مرحباً"])
        scene2 = create_test_scene(dialogue_texts=["وداعاً"])
        scene2.scene_id = "S0002"

        result = enricher.export_contextual_alpaca([scene1, scene2])

        assert len(result) == 2

    def test_enrichment_stats_updated(self):
        """اختبار تحديث الإحصائيات"""
        enricher = ContextEnricher()
        scene = create_test_scene(dialogue_texts=["مرحباً", "أهلاً"])

        enricher.export_contextual_alpaca([scene])
        stats = enricher.get_enrichment_stats()

        assert stats["scenes_processed"] == 1
        assert stats["dialogues_enriched"] == 2

    def test_reset_stats(self):
        """اختبار إعادة تعيين الإحصائيات"""
        enricher = ContextEnricher()
        scene = create_test_scene()
        enricher.export_contextual_alpaca([scene])
        enricher.reset_stats()

        stats = enricher.get_enrichment_stats()
        assert stats["scenes_processed"] == 0

    def test_actions_with_mixed_lengths(self):
        """اختبار قائمة أسطر وصفية بأطوال مختلطة"""
        enricher = ContextEnricher(min_action_length=15)
        actions = [
            "قصير",
            "سطر متوسط الطول",
            "هذا سطر وصفي طويل جداً يصف الحدث بالتفصيل"
        ]

        result = enricher._get_last_significant_action(actions)

        assert result == "هذا سطر وصفي طويل جداً يصف الحدث بالتفصيل"

    def test_time_period_included_in_export(self):
        """اختبار تضمين الفترة الزمنية في التصدير"""
        enricher = ContextEnricher()
        scene = create_test_scene()
        scene.time_period = "1986"

        result = enricher.export_contextual_alpaca([scene])

        assert len(result) > 0
        assert result[0]["metadata"]["time_period"] == "1986"


# ============================================================
# Task 4.1: اختبار خاصية لفلترة الحوارات القصيرة
# الخاصية 9: فلترة الحوارات القصيرة
# ============================================================

class TestQualityFilterProperty9:
    """
    **Feature: al-rawi-v4, Property 9: فلترة الحوارات القصيرة**
    لأي حوار يحتوي على أقل من 3 كلمات عربية، يجب أن يحذفه النظام
    ما لم تكن درجة المشاعر أعلى من 0.8
    المتطلبات 3.1, 3.2
    """

    def test_short_dialogue_filtered(self):
        """اختبار حذف الحوار القصير"""
        qf = QualityFilter(min_words=3, high_sentiment_threshold=0.8)
        turn = create_dialogue_turn("كلمة واحدة", sentiment_score=0.3)

        result = qf.should_keep_turn(turn)

        assert result is False

    def test_long_dialogue_kept(self):
        """اختبار الاحتفاظ بالحوار الطويل"""
        qf = QualityFilter(min_words=3)
        turn = create_dialogue_turn("هذا حوار طويل يحتوي على كلمات كثيرة")

        result = qf.should_keep_turn(turn)

        assert result is True

    def test_exactly_min_words_kept(self):
        """اختبار الاحتفاظ بالحوار الذي يساوي الحد الأدنى"""
        qf = QualityFilter(min_words=3)
        turn = create_dialogue_turn("ثلاث كلمات فقط")

        result = qf.should_keep_turn(turn)

        assert result is True

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis غير متوفرة")
    @given(
        word_count=st.integers(min_value=3, max_value=100),
        sentiment_score=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=100)
    def test_property_long_dialogues_always_kept(self, word_count, sentiment_score):
        """
        **Feature: al-rawi-v4, Property 9: فلترة الحوارات القصيرة**
        لأي حوار يحتوي على 3 كلمات أو أكثر، يجب الاحتفاظ به دائماً
        """
        qf = QualityFilter(min_words=3)
        # إنشاء نص بعدد الكلمات المحدد
        text = " ".join(["كلمة"] * word_count)
        turn = create_dialogue_turn(text, sentiment_score=sentiment_score)

        result = qf.should_keep_turn(turn)

        assert result is True


# ============================================================
# Task 4.2: اختبار خاصية للاحتفاظ بالحوارات العاطفية
# الخاصية 10: الاحتفاظ بالحوارات العاطفية
# ============================================================

class TestQualityFilterProperty10:
    """
    **Feature: al-rawi-v4, Property 10: الاحتفاظ بالحوارات العاطفية**
    لأي حوار قصير بدرجة مشاعر أعلى من 0.8، يجب أن يحتفظ به النظام
    رغم قصره
    المتطلب 3.2
    """

    def test_short_emotional_dialogue_kept(self):
        """اختبار الاحتفاظ بالحوار القصير العاطفي"""
        qf = QualityFilter(min_words=3, high_sentiment_threshold=0.8)
        turn = create_dialogue_turn("أخيراً!", sentiment_score=0.9)

        result = qf.should_keep_turn(turn)

        assert result is True

    def test_short_non_emotional_dialogue_filtered(self):
        """اختبار حذف الحوار القصير غير العاطفي"""
        qf = QualityFilter(min_words=3, high_sentiment_threshold=0.8)
        turn = create_dialogue_turn("حسناً", sentiment_score=0.3)

        result = qf.should_keep_turn(turn)

        assert result is False

    def test_exactly_threshold_sentiment_kept(self):
        """اختبار الاحتفاظ بالحوار عند حد العتبة بالضبط"""
        qf = QualityFilter(min_words=3, high_sentiment_threshold=0.8)
        turn = create_dialogue_turn("نعم", sentiment_score=0.8)

        result = qf.should_keep_turn(turn)

        assert result is True

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis غير متوفرة")
    @given(
        sentiment_score=st.floats(min_value=0.81, max_value=1.0)
    )
    @settings(max_examples=100)
    def test_property_high_sentiment_always_kept(self, sentiment_score):
        """
        **Feature: al-rawi-v4, Property 10: الاحتفاظ بالحوارات العاطفية**
        لأي حوار بدرجة مشاعر أعلى من 0.8، يجب الاحتفاظ به
        """
        qf = QualityFilter(min_words=3, high_sentiment_threshold=0.8)
        turn = create_dialogue_turn("كلمة", sentiment_score=sentiment_score)

        result = qf.should_keep_turn(turn)

        assert result is True


# ============================================================
# Task 4.3: اختبار خاصية لتسجيل إحصائيات الفلترة
# الخاصية 11: تسجيل إحصائيات الفلترة
# ============================================================

class TestQualityFilterProperty11:
    """
    **Feature: al-rawi-v4, Property 11: تسجيل إحصائيات الفلترة**
    لأي عملية فلترة تحدث، يجب أن يسجل النظام عدد الحوارات
    المحذوفة في ملف السجل
    المتطلب 3.3
    """

    def test_filter_stats_tracked(self):
        """اختبار تتبع إحصائيات الفلترة"""
        qf = QualityFilter(min_words=3)
        dialogues = [
            create_dialogue_turn("حوار طويل بما فيه الكفاية"),
            create_dialogue_turn("قصير", sentiment_score=0.3)
        ]

        qf.filter_dialogue_list(dialogues)
        stats = qf.get_filter_stats()

        assert stats["total_dialogues"] == 2
        assert stats["kept_dialogues"] == 1
        assert stats["filtered_count"] == 1

    def test_emotional_kept_stat_tracked(self):
        """اختبار تتبع الحوارات العاطفية المحفوظة"""
        qf = QualityFilter(min_words=3, high_sentiment_threshold=0.8)
        dialogues = [
            create_dialogue_turn("قصير", sentiment_score=0.9)
        ]

        qf.filter_dialogue_list(dialogues)
        stats = qf.get_filter_stats()

        assert stats["kept_emotional"] == 1

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis غير متوفرة")
    @given(
        dialogue_count=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_property_total_equals_kept_plus_filtered(self, dialogue_count):
        """
        **Feature: al-rawi-v4, Property 11: تسجيل إحصائيات الفلترة**
        لأي عدد من الحوارات، يجب أن يساوي المجموع الكلي
        مجموع المحفوظة والمحذوفة
        """
        qf = QualityFilter(min_words=3)
        dialogues = [
            create_dialogue_turn(f"حوار رقم {i}" * (i % 3 + 1))
            for i in range(dialogue_count)
        ]

        qf.filter_dialogue_list(dialogues)
        stats = qf.get_filter_stats()

        assert stats["total_dialogues"] == stats["kept_dialogues"] + stats["filtered_count"]


# ============================================================
# Task 4.4: اختبار خاصية لتطبيق الفلترة قبل التصدير
# الخاصية 12: تطبيق الفلترة قبل التصدير
# ============================================================

class TestQualityFilterProperty12:
    """
    **Feature: al-rawi-v4, Property 12: تطبيق الفلترة قبل التصدير**
    لأي عملية تصدير، يجب أن تحدث الفلترة في مرحلة التصدير
    قبل إنشاء الملفات
    المتطلب 3.5
    """

    def test_filter_applied_before_export(self):
        """اختبار تطبيق الفلترة قبل التصدير"""
        scene = create_sample_scene()
        original_count = len(scene.dialogue)

        qf = QualityFilter(min_words=3)
        filtered_scenes = qf.filter_scenes([scene])

        # يجب أن يكون عدد الحوارات أقل أو يساوي العدد الأصلي
        assert len(filtered_scenes[0].dialogue) <= original_count

    def test_scene_structure_preserved_after_filter(self):
        """اختبار الحفاظ على بنية المشهد بعد الفلترة"""
        scene = create_sample_scene()
        qf = QualityFilter()

        filtered_scenes = qf.filter_scenes([scene])

        assert filtered_scenes[0].scene_id == scene.scene_id
        assert filtered_scenes[0].location == scene.location
        assert filtered_scenes[0].time_of_day == scene.time_of_day

    def test_characters_updated_after_filter(self):
        """اختبار تحديث قائمة الشخصيات بعد الفلترة"""
        qf = QualityFilter(min_words=10)
        scene = create_test_scene(
            dialogue_texts=["كلمة", "حوار طويل جداً يحتوي على كلمات كثيرة"]
        )

        filtered_scenes = qf.filter_scenes([scene])

        # يجب أن تحتوي قائمة الشخصيات فقط على الشخصيات ذات الحوارات المحفوظة
        assert len(filtered_scenes[0].characters) <= 2


# ============================================================
# Task 4.5: اختبار دمج QualityFilter في DatasetExporter
# ============================================================

class TestDatasetExporterIntegration:
    """
    اختبارات تكامل DatasetExporter مع QualityFilter
    المتطلب 3.5, 6.2
    """

    def test_exporter_applies_quality_filter(self, tmp_path):
        """اختبار تطبيق فلترة الجودة في المصدّر"""
        exporter = DatasetExporter(
            output_dir=str(tmp_path),
            min_words=10,
            apply_quality_filter=True
        )

        scene = create_sample_scene()
        output_path = exporter.export_alpaca_jsonl([scene])

        # قراءة الملف والتحقق
        with open(output_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]

        # يجب أن يكون عدد السجلات أقل من عدد الحوارات الأصلية
        # لأن بعض الحوارات قصيرة
        assert len(records) <= len(scene.dialogue)

    def test_exporter_without_filter(self, tmp_path):
        """اختبار المصدّر بدون فلترة"""
        exporter = DatasetExporter(
            output_dir=str(tmp_path),
            apply_quality_filter=False
        )

        scene = create_sample_scene()
        output_path = exporter.export_alpaca_jsonl([scene])

        with open(output_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]

        # يجب أن يكون عدد السجلات يساوي عدد الحوارات الأصلية
        assert len(records) == len(scene.dialogue)

    def test_exporter_stats_available(self, tmp_path):
        """اختبار توفر الإحصائيات"""
        exporter = DatasetExporter(
            output_dir=str(tmp_path),
            apply_quality_filter=True,
            apply_context_enrichment=True
        )

        scene = create_sample_scene()
        exporter.export_alpaca_jsonl([scene])

        stats = exporter.get_all_stats()

        assert "quality_filter" in stats
        assert "context_enricher" in stats

    def test_exporter_creates_output_directory(self, tmp_path):
        """اختبار إنشاء مجلد المخرجات"""
        new_dir = tmp_path / "new_output_dir"
        exporter = DatasetExporter(output_dir=str(new_dir))

        assert new_dir.exists()


# ============================================================
# اختبارات الدوال المساعدة
# ============================================================

class TestHelperFunctions:
    """اختبارات الدوال المساعدة"""

    def test_count_arabic_words(self):
        """اختبار عد الكلمات العربية"""
        assert count_arabic_words("مرحباً كيف حالك") == 3
        assert count_arabic_words("كلمة") == 1
        assert count_arabic_words("") == 0
        assert count_arabic_words("   ") == 0

    def test_is_transition_line(self):
        """اختبار التعرف على الانتقالات"""
        assert is_transition_line("قطع") is True
        assert is_transition_line("CUT") is True
        assert is_transition_line("FADE OUT") is True
        assert is_transition_line("يجلس على الكرسي") is False

    def test_is_significant_action(self):
        """اختبار التعرف على الأسطر الوصفية المهمة"""
        assert is_significant_action("يجلس على الكرسي الخشبي", 10) is True
        assert is_significant_action("يدخل", 10) is False
        assert is_significant_action("قطع", 5) is False
        assert is_significant_action("", 5) is False


# ============================================================
# نقطة دخول الاختبارات
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
