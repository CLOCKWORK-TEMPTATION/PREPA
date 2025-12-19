#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اختبارات وحدة توحيد الكيانات (Entity Canonicalizer Tests)
==========================================================

هذا الملف يحتوي على اختبارات الخصائص (Property-Based Tests) واختبارات الوحدة
لوحدة توحيد الكيانات.

يستخدم مكتبة Hypothesis لاختبارات الخصائص.
"""

import pytest
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from entity_canonicalizer import (
    EntityCanonicalizer,
    normalize_arabic_text,
    count_arabic_words,
    canonicalize_scenes,
    SIMILARITY_AVAILABLE,
    MergeLogEntry
)


# ---------------------------------------------------------
# نماذج بيانات للاختبار
# ---------------------------------------------------------
@dataclass
class MockDialogueTurn:
    """نموذج حوار للاختبار"""
    scene_id: str
    turn_id: int
    speaker: str
    text: str
    original_speaker: str = ""
    sentiment: str = "unknown"
    sentiment_score: float = 0.0


@dataclass
class MockScene:
    """نموذج مشهد للاختبار"""
    scene_id: str
    scene_number: int
    heading: str
    location: str
    time_of_day: str
    int_ext: str
    actions: List[str] = field(default_factory=list)
    dialogue: List[MockDialogueTurn] = field(default_factory=list)
    characters: List[str] = field(default_factory=list)
    full_text: str = ""


# ---------------------------------------------------------
# Strategies للاختبارات (مولدات البيانات)
# ---------------------------------------------------------

# استراتيجية لتوليد أسماء عربية
arabic_chars = "أبتثجحخدذرزسشصضطظعغفقكلمنهويئءةى"
arabic_name_strategy = st.text(
    alphabet=arabic_chars + " ",
    min_size=2,
    max_size=20
).filter(lambda x: len(x.strip()) >= 2)

# استراتيجية لتوليد أسماء مشابهة (اسم + لاحقة)
@st.composite
def similar_names_strategy(draw):
    """توليد زوج من الأسماء المتشابهة"""
    base_name = draw(st.text(alphabet=arabic_chars, min_size=3, max_size=10))
    assume(len(base_name.strip()) >= 3)

    suffix = draw(st.text(alphabet=arabic_chars + " ", min_size=0, max_size=10))

    name1 = base_name.strip()
    name2 = (base_name + " " + suffix).strip()

    return name1, name2

# استراتيجية لتوليد قائمة أسماء
names_list_strategy = st.lists(
    arabic_name_strategy,
    min_size=2,
    max_size=20
)

# استراتيجية لتوليد حوار
@st.composite
def dialogue_turn_strategy(draw):
    """توليد حوار للاختبار"""
    scene_id = draw(st.text(alphabet="S0123456789", min_size=5, max_size=5))
    turn_id = draw(st.integers(min_value=1, max_value=100))
    speaker = draw(arabic_name_strategy)
    text = draw(st.text(alphabet=arabic_chars + " .,!?", min_size=5, max_size=200))

    return MockDialogueTurn(
        scene_id=scene_id,
        turn_id=turn_id,
        speaker=speaker,
        text=text
    )

# استراتيجية لتوليد مشهد
@st.composite
def scene_strategy(draw):
    """توليد مشهد للاختبار"""
    scene_id = draw(st.text(alphabet="S0123456789", min_size=5, max_size=5))
    scene_number = draw(st.integers(min_value=1, max_value=100))
    dialogue_count = draw(st.integers(min_value=0, max_value=10))

    dialogue = []
    for i in range(dialogue_count):
        turn = draw(dialogue_turn_strategy())
        turn.scene_id = scene_id
        turn.turn_id = i + 1
        dialogue.append(turn)

    characters = list(set(t.speaker for t in dialogue if t.speaker))

    return MockScene(
        scene_id=scene_id,
        scene_number=scene_number,
        heading=f"مشهد {scene_number}",
        location=draw(st.text(alphabet=arabic_chars + " ", min_size=3, max_size=30)),
        time_of_day=draw(st.sampled_from(["ليل", "نهار", "صباح", "مساء"])),
        int_ext=draw(st.sampled_from(["داخلي", "خارجي"])),
        dialogue=dialogue,
        characters=characters
    )


# ---------------------------------------------------------
# اختبارات الخصائص (Property-Based Tests)
# ---------------------------------------------------------

class TestSimilarityCalculation:
    """
    اختبارات خاصية حساب التشابه

    **Feature: al-rawi-v4, Property 1: حساب التشابه للأسماء المتشابهة**
    """

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_similarity_with_itself_is_one(self, name):
        """
        **Feature: al-rawi-v4, Property 1: حساب التشابه للأسماء المتشابهة**

        التشابه بين اسم ونفسه يجب أن يكون 1.0
        **تتحقق من: المتطلبات 1.1, 1.5**
        """
        assume(len(name.strip()) > 0)
        canonicalizer = EntityCanonicalizer()
        similarity = canonicalizer.calculate_similarity(name, name)
        assert similarity == 1.0

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(st.text(min_size=1, max_size=30), st.text(min_size=1, max_size=30))
    @settings(max_examples=100)
    def test_similarity_is_symmetric(self, name1, name2):
        """
        **Feature: al-rawi-v4, Property 1: حساب التشابه للأسماء المتشابهة**

        التشابه يجب أن يكون متماثل: sim(a, b) == sim(b, a)
        **تتحقق من: المتطلبات 1.1, 1.5**
        """
        assume(len(name1.strip()) > 0 and len(name2.strip()) > 0)
        canonicalizer = EntityCanonicalizer()

        sim_ab = canonicalizer.calculate_similarity(name1, name2)
        sim_ba = canonicalizer.calculate_similarity(name2, name1)

        assert abs(sim_ab - sim_ba) < 0.001  # السماح بفرق بسيط للتقريب

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(st.text(min_size=1, max_size=30), st.text(min_size=1, max_size=30))
    @settings(max_examples=100)
    def test_similarity_is_between_0_and_1(self, name1, name2):
        """
        **Feature: al-rawi-v4, Property 1: حساب التشابه للأسماء المتشابهة**

        نسبة التشابه يجب أن تكون بين 0 و 1
        **تتحقق من: المتطلبات 1.1, 1.5**
        """
        assume(len(name1.strip()) > 0 and len(name2.strip()) > 0)
        canonicalizer = EntityCanonicalizer()

        similarity = canonicalizer.calculate_similarity(name1, name2)

        assert 0.0 <= similarity <= 1.0

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(similar_names_strategy())
    @settings(max_examples=100)
    def test_similar_names_have_high_similarity(self, names):
        """
        **Feature: al-rawi-v4, Property 1: حساب التشابه للأسماء المتشابهة**

        الأسماء المتشابهة (اسم + لاحقة) يجب أن يكون لها تشابه عالي
        **تتحقق من: المتطلبات 1.1, 1.5**
        """
        name1, name2 = names
        assume(len(name1) >= 3 and len(name2) >= 3)

        canonicalizer = EntityCanonicalizer()
        similarity = canonicalizer.calculate_similarity(name1, name2)

        # إذا كان اسم مضمن في الآخر، التشابه يجب أن يكون عالي
        if name1 in name2 or name2 in name1:
            assert similarity >= 0.5


class TestNameBinding:
    """
    اختبارات خاصية ربط الأسماء

    **Feature: al-rawi-v4, Property 2: ربط الأسماء عالية التشابه**
    """

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(names_list_strategy)
    @settings(max_examples=100)
    def test_canonical_map_values_are_from_input(self, names):
        """
        **Feature: al-rawi-v4, Property 2: ربط الأسماء عالية التشابه**

        كل قيمة في قاموس التطبيع يجب أن تكون من الأسماء الأصلية
        **تتحقق من: المتطلبات 1.2**
        """
        assume(len(names) >= 2)
        assume(all(len(n.strip()) >= 2 for n in names))

        canonicalizer = EntityCanonicalizer(similarity_threshold=0.85)
        canonical_map = canonicalizer.build_canonical_map_from_names(names)

        unique_names = set(n.strip() for n in names if n.strip())

        for original, canonical in canonical_map.items():
            assert canonical in unique_names, f"الاسم الكانوني '{canonical}' ليس من الأسماء الأصلية"

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(names_list_strategy)
    @settings(max_examples=100)
    def test_canonical_name_is_not_mapped(self, names):
        """
        **Feature: al-rawi-v4, Property 2: ربط الأسماء عالية التشابه**

        الاسم الكانوني لا يجب أن يكون مفتاحاً في القاموس (لا يُربط لاسم آخر)
        **تتحقق من: المتطلبات 1.2**
        """
        assume(len(names) >= 2)
        assume(all(len(n.strip()) >= 2 for n in names))

        canonicalizer = EntityCanonicalizer(similarity_threshold=0.85)
        canonical_map = canonicalizer.build_canonical_map_from_names(names)

        canonical_names = set(canonical_map.values())

        for canonical in canonical_names:
            assert canonical not in canonical_map, f"الاسم الكانوني '{canonical}' مربوط لاسم آخر"

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=100)
    def test_threshold_is_respected(self, threshold):
        """
        **Feature: al-rawi-v4, Property 2: ربط الأسماء عالية التشابه**

        عتبة التشابه يجب أن تُحترم في عمليات الدمج
        **تتحقق من: المتطلبات 1.2**
        """
        # أسماء اختبار ثابتة
        names = ["محمد", "محمود", "أحمد", "أحمد السيد"]

        canonicalizer = EntityCanonicalizer(similarity_threshold=threshold)
        canonical_map = canonicalizer.build_canonical_map_from_names(names)

        # التحقق من أن كل عملية دمج تحترم العتبة
        for entry in canonicalizer.merge_log:
            assert entry.similarity_score >= threshold


class TestComprehensiveNormalization:
    """
    اختبارات التطبيع الشامل

    **Feature: al-rawi-v4, Property 3: تطبيق التطبيع الشامل**
    """

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(st.lists(scene_strategy(), min_size=1, max_size=5))
    @settings(max_examples=50)
    def test_all_speakers_are_normalized(self, scenes):
        """
        **Feature: al-rawi-v4, Property 3: تطبيق التطبيع الشامل**

        بعد التطبيع، كل متحدث يجب أن يكون اسمه كانونياً
        **تتحقق من: المتطلبات 1.3**
        """
        assume(any(len(s.dialogue) > 0 for s in scenes))

        canonicalizer = EntityCanonicalizer(similarity_threshold=0.85)
        canonicalizer.build_canonical_map(scenes)
        normalized_scenes = canonicalizer.apply_normalization(scenes)

        for scene in normalized_scenes:
            for turn in scene.dialogue:
                if turn.speaker:
                    # المتحدث يجب أن يكون إما اسم كانوني أو غير موجود في القاموس
                    assert turn.speaker not in canonicalizer.canonical_map

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(st.lists(scene_strategy(), min_size=1, max_size=5))
    @settings(max_examples=50)
    def test_normalization_is_idempotent(self, scenes):
        """
        **Feature: al-rawi-v4, Property 3: تطبيق التطبيع الشامل**

        تطبيق التطبيع مرتين يعطي نفس النتيجة
        **تتحقق من: المتطلبات 1.3**
        """
        assume(any(len(s.dialogue) > 0 for s in scenes))

        canonicalizer = EntityCanonicalizer(similarity_threshold=0.85)
        canonicalizer.build_canonical_map(scenes)

        # التطبيع الأول
        normalized_once = canonicalizer.apply_normalization(scenes)
        speakers_once = [
            t.speaker for s in normalized_once for t in s.dialogue
        ]

        # التطبيع الثاني
        normalized_twice = canonicalizer.apply_normalization(normalized_once)
        speakers_twice = [
            t.speaker for s in normalized_twice for t in s.dialogue
        ]

        assert speakers_once == speakers_twice


class TestMergeDocumentation:
    """
    اختبارات توثيق عمليات الدمج

    **Feature: al-rawi-v4, Property 4: توثيق عمليات الدمج**
    """

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(names_list_strategy)
    @settings(max_examples=100)
    def test_merge_log_contains_all_merges(self, names):
        """
        **Feature: al-rawi-v4, Property 4: توثيق عمليات الدمج**

        سجل الدمج يجب أن يحتوي على جميع عمليات الدمج
        **تتحقق من: المتطلبات 1.4**
        """
        assume(len(names) >= 2)
        assume(all(len(n.strip()) >= 2 for n in names))

        canonicalizer = EntityCanonicalizer(similarity_threshold=0.85)
        canonical_map = canonicalizer.build_canonical_map_from_names(names)

        # عدد عمليات الدمج يجب أن يساوي عدد المفاتيح في القاموس
        assert len(canonicalizer.merge_log) == len(canonical_map)

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(names_list_strategy)
    @settings(max_examples=50)
    def test_merge_log_can_be_exported(self, names):
        """
        **Feature: al-rawi-v4, Property 4: توثيق عمليات الدمج**

        سجل الدمج يجب أن يكون قابلاً للتصدير لملف
        **تتحقق من: المتطلبات 1.4**
        """
        assume(len(names) >= 2)
        assume(all(len(n.strip()) >= 2 for n in names))

        canonicalizer = EntityCanonicalizer(similarity_threshold=0.85)
        canonicalizer.build_canonical_map_from_names(names)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "merge_log.json"
            canonicalizer.export_merge_log(log_path)

            if canonicalizer.merge_log:
                assert log_path.exists(), "ملف سجل الدمج يجب أن يُنشأ"

                with open(log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

                assert "عمليات_الدمج" in log_data
                assert len(log_data["عمليات_الدمج"]) == len(canonicalizer.merge_log)

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    @given(names_list_strategy)
    @settings(max_examples=100)
    def test_merge_log_entries_have_required_fields(self, names):
        """
        **Feature: al-rawi-v4, Property 4: توثيق عمليات الدمج**

        كل سجل دمج يجب أن يحتوي على الحقول المطلوبة
        **تتحقق من: المتطلبات 1.4**
        """
        assume(len(names) >= 2)
        assume(all(len(n.strip()) >= 2 for n in names))

        canonicalizer = EntityCanonicalizer(similarity_threshold=0.85)
        canonicalizer.build_canonical_map_from_names(names)

        for entry in canonicalizer.merge_log:
            assert hasattr(entry, 'original_name')
            assert hasattr(entry, 'canonical_name')
            assert hasattr(entry, 'similarity_score')
            assert hasattr(entry, 'original_count')
            assert hasattr(entry, 'merge_reason')


# ---------------------------------------------------------
# اختبارات الوحدة (Unit Tests)
# ---------------------------------------------------------

class TestUnitNormalization:
    """اختبارات وحدة لتطبيع النصوص العربية"""

    def test_normalize_arabic_removes_diacritics(self):
        """اختبار إزالة التشكيل"""
        text = "مُحَمَّد"
        normalized = normalize_arabic_text(text)
        assert "ُ" not in normalized
        assert "َ" not in normalized
        assert "ّ" not in normalized

    def test_normalize_arabic_unifies_alef(self):
        """اختبار توحيد الألف"""
        assert normalize_arabic_text("أحمد") == normalize_arabic_text("احمد")
        assert normalize_arabic_text("إسلام") == normalize_arabic_text("اسلام")
        assert normalize_arabic_text("آمال") == normalize_arabic_text("امال")

    def test_normalize_arabic_unifies_yaa(self):
        """اختبار توحيد الياء"""
        assert normalize_arabic_text("على") == normalize_arabic_text("علي")

    def test_normalize_arabic_handles_empty(self):
        """اختبار النص الفارغ"""
        assert normalize_arabic_text("") == ""
        assert normalize_arabic_text(None) == ""


class TestUnitEntityCanonicalizer:
    """اختبارات وحدة لفئة EntityCanonicalizer"""

    def test_init_with_valid_threshold(self):
        """اختبار التهيئة بعتبة صالحة"""
        canonicalizer = EntityCanonicalizer(similarity_threshold=0.90)
        assert canonicalizer.threshold == 0.90

    def test_init_with_invalid_threshold_raises(self):
        """اختبار التهيئة بعتبة غير صالحة"""
        with pytest.raises(ValueError):
            EntityCanonicalizer(similarity_threshold=1.5)

        with pytest.raises(ValueError):
            EntityCanonicalizer(similarity_threshold=-0.1)

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    def test_calculate_similarity_identical_names(self):
        """اختبار التشابه بين اسمين متطابقين"""
        canonicalizer = EntityCanonicalizer()
        assert canonicalizer.calculate_similarity("رأفت", "رأفت") == 1.0

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    def test_calculate_similarity_different_names(self):
        """اختبار التشابه بين اسمين مختلفين"""
        canonicalizer = EntityCanonicalizer()
        similarity = canonicalizer.calculate_similarity("رأفت", "محمد")
        assert similarity < 0.5

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    def test_calculate_similarity_similar_names(self):
        """اختبار التشابه بين اسمين متشابهين"""
        canonicalizer = EntityCanonicalizer()
        similarity = canonicalizer.calculate_similarity("رأفت", "رأفت الهجان")
        assert similarity >= 0.5

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    def test_build_canonical_map_example(self):
        """اختبار بناء قاموس التطبيع مع مثال محدد"""
        names = [
            "رأفت", "رأفت", "رأفت", "رأفت",  # 4 مرات
            "رأفت الهجان",  # 1 مرة
        ]

        canonicalizer = EntityCanonicalizer(similarity_threshold=0.80)
        canonical_map = canonicalizer.build_canonical_map_from_names(names)

        # "رأفت" يجب أن يكون الكانوني لأنه الأكثر تكراراً
        if "رأفت الهجان" in canonical_map:
            assert canonical_map["رأفت الهجان"] == "رأفت"

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    def test_normalize_character_name(self):
        """اختبار تطبيع اسم شخصية"""
        canonicalizer = EntityCanonicalizer()
        canonicalizer.canonical_map = {"رأفت الهجان": "رأفت"}

        assert canonicalizer.normalize_character_name("رأفت الهجان") == "رأفت"
        assert canonicalizer.normalize_character_name("محمد") == "محمد"  # غير موجود في القاموس

    def test_empty_name_returns_empty(self):
        """اختبار أن الاسم الفارغ يعود فارغاً"""
        canonicalizer = EntityCanonicalizer()
        assert canonicalizer.normalize_character_name("") == ""
        assert canonicalizer.normalize_character_name("   ") == ""

    def test_get_statistics(self):
        """اختبار الحصول على الإحصائيات"""
        canonicalizer = EntityCanonicalizer(similarity_threshold=0.85)
        stats = canonicalizer.get_statistics()

        assert "إجمالي_الأسماء" in stats
        assert "الأسماء_المدمجة" in stats
        assert "عتبة_التشابه" in stats
        assert stats["عتبة_التشابه"] == 0.85


class TestIntegration:
    """اختبارات التكامل"""

    @pytest.mark.skipif(not SIMILARITY_AVAILABLE, reason="لا توجد مكتبات تشابه")
    def test_full_workflow(self):
        """اختبار سير العمل الكامل"""
        # إنشاء مشاهد اختبار
        scenes = [
            MockScene(
                scene_id="S0001",
                scene_number=1,
                heading="مشهد 1",
                location="غرفة المعيشة",
                time_of_day="نهار",
                int_ext="داخلي",
                dialogue=[
                    MockDialogueTurn("S0001", 1, "رأفت", "مرحباً"),
                    MockDialogueTurn("S0001", 2, "رأفت الهجان", "أهلاً بك"),
                    MockDialogueTurn("S0001", 3, "رأفت", "كيف حالك؟"),
                ],
                characters=["رأفت", "رأفت الهجان"]
            ),
            MockScene(
                scene_id="S0002",
                scene_number=2,
                heading="مشهد 2",
                location="المكتب",
                time_of_day="ليل",
                int_ext="داخلي",
                dialogue=[
                    MockDialogueTurn("S0002", 1, "رأفت", "أنا هنا"),
                    MockDialogueTurn("S0002", 2, "محمد", "وأنا أيضاً"),
                ],
                characters=["رأفت", "محمد"]
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "merge_log.json"

            normalized_scenes, canonical_map, stats = canonicalize_scenes(
                scenes,
                similarity_threshold=0.80,
                merge_log_path=log_path
            )

            # التحقق من النتائج
            assert len(normalized_scenes) == 2
            assert isinstance(canonical_map, dict)
            assert isinstance(stats, dict)

            # التحقق من أن جميع المتحدثين موحدين
            all_speakers = [
                t.speaker
                for s in normalized_scenes
                for t in s.dialogue
            ]

            # لا يجب أن يكون هناك متحدث في قاموس التطبيع
            for speaker in all_speakers:
                assert speaker not in canonical_map


# ---------------------------------------------------------
# تشغيل الاختبارات
# ---------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
