#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
وحدة توحيد الكيانات (Entity Canonicalizer)
===========================================

هذه الوحدة مسؤولة عن توحيد أسماء الشخصيات المتشابهة في السيناريوهات.
تستخدم خوارزميات حساب التشابه لربط الأسماء المتقاربة بالاسم الأكثر استخداماً.

المتطلبات المحققة: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# إعداد التسجيل
logger = logging.getLogger(__name__)

# --- استيراد آمن للمكتبات ---
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
    logger.info("تم تحميل مكتبة rapidfuzz بنجاح")
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning("مكتبة rapidfuzz غير متوفرة - محاولة استخدام difflib")

try:
    import difflib
    DIFFLIB_AVAILABLE = True
except ImportError:
    DIFFLIB_AVAILABLE = False
    logger.warning("مكتبة difflib غير متوفرة")

# التحقق من توفر مكتبات حساب التشابه
SIMILARITY_AVAILABLE = RAPIDFUZZ_AVAILABLE or DIFFLIB_AVAILABLE


# ---------------------------------------------------------
# دوال مساعدة لتطبيع النصوص العربية
# ---------------------------------------------------------
def normalize_arabic_text(text: str) -> str:
    """
    تطبيع النص العربي لتحسين المقارنة

    - إزالة التشكيل
    - توحيد الهمزات
    - توحيد الياء والألف المقصورة
    - إزالة التطويل
    """
    if not text:
        return ""

    # إزالة التشكيل
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)

    # إزالة التطويل (ـ)
    text = re.sub(r'\u0640+', '', text)

    # توحيد الهمزات
    text = re.sub(r'[إأآا]', 'ا', text)

    # توحيد الياء والألف المقصورة
    text = re.sub(r'[يى]', 'ي', text)

    # توحيد التاء المربوطة والهاء
    text = re.sub(r'ة', 'ه', text)

    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def count_arabic_words(text: str) -> int:
    """
    عد الكلمات العربية في النص
    """
    if not text:
        return 0
    return len(re.findall(r'[\u0600-\u06FF]+', text))


# ---------------------------------------------------------
# فئة MergeLogEntry لتوثيق عمليات الدمج
# ---------------------------------------------------------
@dataclass
class MergeLogEntry:
    """
    تسجيل عملية دمج اسم واحدة
    """
    original_name: str
    canonical_name: str
    similarity_score: float
    original_count: int
    merge_reason: str


# ---------------------------------------------------------
# فئة EntityCanonicalizer الرئيسية
# ---------------------------------------------------------
class EntityCanonicalizer:
    """
    مسؤول عن توحيد أسماء الشخصيات المتشابهة

    يستخدم خوارزمية Levenshtein (المسافة الليفنشتاين) لحساب التشابه
    بين الأسماء، ويربط الأسماء المتشابهة بالاسم الأكثر تكراراً.

    Args:
        similarity_threshold: نسبة التشابه المطلوبة للدمج (افتراضي: 85%)

    Attributes:
        threshold: عتبة التشابه
        canonical_map: قاموس يربط الأسماء بأسمائها الكانونية
        merge_log: سجل عمليات الدمج
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        تهيئة محلل توحيد الكيانات

        Args:
            similarity_threshold: نسبة التشابه المطلوبة للدمج (0.0 - 1.0)
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("نسبة التشابه يجب أن تكون بين 0.0 و 1.0")

        self.threshold = similarity_threshold
        self.canonical_map: Dict[str, str] = {}
        self.merge_log: List[MergeLogEntry] = []
        self._name_counts: Dict[str, int] = {}

        # التحقق من توفر مكتبات حساب التشابه
        if not SIMILARITY_AVAILABLE:
            logger.error("لا توجد مكتبات حساب التشابه - تخطي توحيد الكيانات")

    def calculate_similarity(self, name1: str, name2: str) -> float:
        """
        حساب نسبة التشابه بين اسمين

        يستخدم rapidfuzz إذا كان متوفراً، وإلا يستخدم difflib.

        Args:
            name1: الاسم الأول
            name2: الاسم الثاني

        Returns:
            نسبة التشابه (0.0 - 1.0)
        """
        if not name1 or not name2:
            return 0.0

        # تطبيع الأسماء قبل المقارنة
        norm1 = normalize_arabic_text(name1)
        norm2 = normalize_arabic_text(name2)

        if norm1 == norm2:
            return 1.0

        if RAPIDFUZZ_AVAILABLE:
            # rapidfuzz يعيد نسبة من 0-100
            return fuzz.ratio(norm1, norm2) / 100.0
        elif DIFFLIB_AVAILABLE:
            return difflib.SequenceMatcher(None, norm1, norm2).ratio()
        else:
            # إذا لم تتوفر أي مكتبة، نستخدم مقارنة بسيطة
            logger.warning("لا توجد مكتبات تشابه - استخدام مقارنة بسيطة")
            return 1.0 if norm1 == norm2 else 0.0

    def _collect_all_names(self, scenes: List[Any]) -> Dict[str, int]:
        """
        جمع جميع أسماء الشخصيات من المشاهد مع عدد تكراراتها

        Args:
            scenes: قائمة المشاهد

        Returns:
            قاموس {اسم: عدد_التكرارات}
        """
        name_counts: Dict[str, int] = Counter()

        for scene in scenes:
            # التحقق من وجود حوارات في المشهد
            dialogue = getattr(scene, 'dialogue', [])
            if not dialogue:
                continue

            for turn in dialogue:
                speaker = getattr(turn, 'speaker', None)
                if speaker and speaker.strip():
                    name_counts[speaker.strip()] += 1

        return dict(name_counts)

    def _find_canonical_name(self, names: List[str], counts: Dict[str, int]) -> str:
        """
        اختيار الاسم الكانوني من مجموعة أسماء متشابهة

        القواعد:
        1. الاسم الأكثر تكراراً
        2. إذا تساوت التكرارات، الاسم الأطول

        Args:
            names: قائمة الأسماء المتشابهة
            counts: قاموس تكرارات الأسماء

        Returns:
            الاسم الكانوني
        """
        if not names:
            return ""

        if len(names) == 1:
            return names[0]

        # ترتيب حسب التكرار (تنازلي) ثم الطول (تنازلي)
        sorted_names = sorted(
            names,
            key=lambda n: (counts.get(n, 0), len(n)),
            reverse=True
        )

        return sorted_names[0]

    def _group_similar_names(self, names: List[str]) -> List[List[str]]:
        """
        تجميع الأسماء المتشابهة في مجموعات

        Args:
            names: قائمة جميع الأسماء

        Returns:
            قائمة من مجموعات الأسماء المتشابهة
        """
        if not names:
            return []

        # نستخدم Union-Find لتجميع الأسماء
        parent = {name: name for name in names}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # حساب التشابه بين كل زوج
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                similarity = self.calculate_similarity(name1, name2)
                if similarity >= self.threshold:
                    union(name1, name2)

        # تجميع الأسماء حسب الجذر
        groups: Dict[str, List[str]] = {}
        for name in names:
            root = find(name)
            if root not in groups:
                groups[root] = []
            groups[root].append(name)

        return list(groups.values())

    def build_canonical_map(self, scenes: List[Any]) -> Dict[str, str]:
        """
        بناء قاموس التطبيع من جميع المشاهد

        الخوارزمية:
        1. جمع جميع الأسماء مع تكراراتها
        2. تجميع الأسماء المتشابهة (نسبة تشابه > العتبة)
        3. اختيار الاسم الكانوني لكل مجموعة
        4. بناء قاموس التطبيع

        Args:
            scenes: قائمة المشاهد

        Returns:
            قاموس يربط الأسماء المتشابهة بالاسم الكانوني
        """
        if not SIMILARITY_AVAILABLE:
            logger.warning("لا توجد مكتبات تشابه - تخطي بناء قاموس التطبيع")
            return {}

        # جمع الأسماء والتكرارات
        self._name_counts = self._collect_all_names(scenes)

        if not self._name_counts:
            logger.info("لا توجد أسماء شخصيات للتوحيد")
            return {}

        all_names = list(self._name_counts.keys())
        logger.info(f"تم العثور على {len(all_names)} اسم شخصية فريد")

        # تجميع الأسماء المتشابهة
        similar_groups = self._group_similar_names(all_names)

        # بناء قاموس التطبيع
        self.canonical_map = {}
        self.merge_log = []

        merged_count = 0
        for group in similar_groups:
            if len(group) < 2:
                continue

            canonical = self._find_canonical_name(group, self._name_counts)

            for name in group:
                if name != canonical:
                    self.canonical_map[name] = canonical
                    merged_count += 1

                    # تسجيل عملية الدمج
                    similarity = self.calculate_similarity(name, canonical)
                    self.merge_log.append(MergeLogEntry(
                        original_name=name,
                        canonical_name=canonical,
                        similarity_score=similarity,
                        original_count=self._name_counts.get(name, 0),
                        merge_reason=f"نسبة التشابه: {similarity:.2%}"
                    ))

        logger.info(f"تم دمج {merged_count} اسم في {len(similar_groups)} مجموعة")

        return self.canonical_map

    def build_canonical_map_from_names(self, names: List[str]) -> Dict[str, str]:
        """
        بناء قاموس التطبيع من قائمة أسماء

        هذه الدالة للاستخدام في الاختبارات

        Args:
            names: قائمة الأسماء

        Returns:
            قاموس التطبيع
        """
        if not SIMILARITY_AVAILABLE:
            return {}

        # حساب التكرارات
        self._name_counts = Counter(names)
        unique_names = list(set(names))

        if not unique_names:
            return {}

        # تجميع الأسماء المتشابهة
        similar_groups = self._group_similar_names(unique_names)

        # بناء القاموس
        self.canonical_map = {}
        self.merge_log = []

        for group in similar_groups:
            if len(group) < 2:
                continue

            canonical = self._find_canonical_name(group, dict(self._name_counts))

            for name in group:
                if name != canonical:
                    self.canonical_map[name] = canonical
                    similarity = self.calculate_similarity(name, canonical)
                    self.merge_log.append(MergeLogEntry(
                        original_name=name,
                        canonical_name=canonical,
                        similarity_score=similarity,
                        original_count=self._name_counts.get(name, 0),
                        merge_reason=f"نسبة التشابه: {similarity:.2%}"
                    ))

        return self.canonical_map

    def normalize_character_name(self, name: str) -> str:
        """
        تطبيع اسم شخصية واحدة

        Args:
            name: الاسم الأصلي

        Returns:
            الاسم الكانوني (أو الاسم الأصلي إذا لم يكن هناك تطبيع)
        """
        if not name:
            return name

        name = name.strip()
        return self.canonical_map.get(name, name)

    def apply_normalization(self, scenes: List[Any]) -> List[Any]:
        """
        تطبيق التطبيع على جميع الحوارات في المشاهد

        يقوم بتحديث اسم المتحدث (speaker) في كل حوار
        ليستخدم الاسم الكانوني

        Args:
            scenes: قائمة المشاهد

        Returns:
            قائمة المشاهد بعد التطبيع
        """
        if not self.canonical_map:
            logger.info("لا يوجد قاموس تطبيع - تخطي التطبيع")
            return scenes

        normalized_count = 0

        for scene in scenes:
            dialogue = getattr(scene, 'dialogue', [])

            for turn in dialogue:
                original_speaker = getattr(turn, 'speaker', None)
                if original_speaker:
                    normalized_speaker = self.normalize_character_name(original_speaker)

                    if normalized_speaker != original_speaker:
                        # حفظ الاسم الأصلي إذا كان الحقل موجوداً
                        if hasattr(turn, 'original_speaker'):
                            turn.original_speaker = original_speaker

                        # تحديث الاسم
                        turn.speaker = normalized_speaker
                        normalized_count += 1

            # تحديث قائمة الشخصيات في المشهد
            if hasattr(scene, 'characters') and hasattr(scene, 'dialogue'):
                scene.characters = list(set(
                    self.normalize_character_name(c)
                    for c in scene.characters
                ))

        logger.info(f"تم تطبيع {normalized_count} حوار")

        return scenes

    def export_merge_log(self, output_path: Path) -> None:
        """
        حفظ سجل عمليات الدمج في ملف JSON

        Args:
            output_path: مسار ملف الإخراج
        """
        if not self.merge_log:
            logger.info("لا توجد عمليات دمج للتصدير")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        log_data = {
            "عتبة_التشابه": self.threshold,
            "عدد_عمليات_الدمج": len(self.merge_log),
            "عمليات_الدمج": [
                {
                    "الاسم_الأصلي": entry.original_name,
                    "الاسم_الكانوني": entry.canonical_name,
                    "نسبة_التشابه": entry.similarity_score,
                    "التكرار_الأصلي": entry.original_count,
                    "السبب": entry.merge_reason
                }
                for entry in self.merge_log
            ]
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            logger.info(f"تم حفظ سجل الدمج في: {output_path}")
        except Exception as e:
            logger.error(f"فشل حفظ سجل الدمج: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        الحصول على إحصائيات التوحيد

        Returns:
            قاموس الإحصائيات
        """
        return {
            "إجمالي_الأسماء": len(self._name_counts),
            "الأسماء_المدمجة": len(self.merge_log),
            "الأسماء_الكانونية": len(set(self.canonical_map.values())),
            "عتبة_التشابه": self.threshold,
            "مكتبة_التشابه": "rapidfuzz" if RAPIDFUZZ_AVAILABLE else "difflib" if DIFFLIB_AVAILABLE else "غير متوفرة"
        }


# ---------------------------------------------------------
# دوال مساعدة للتكامل
# ---------------------------------------------------------
def canonicalize_scenes(
    scenes: List[Any],
    similarity_threshold: float = 0.85,
    merge_log_path: Optional[Path] = None
) -> Tuple[List[Any], Dict[str, str], Dict[str, Any]]:
    """
    دالة مساعدة لتوحيد أسماء الشخصيات في المشاهد

    Args:
        scenes: قائمة المشاهد
        similarity_threshold: عتبة التشابه
        merge_log_path: مسار حفظ سجل الدمج (اختياري)

    Returns:
        tuple من:
        - المشاهد بعد التوحيد
        - قاموس التطبيع
        - إحصائيات التوحيد
    """
    canonicalizer = EntityCanonicalizer(similarity_threshold)

    # بناء القاموس
    canonical_map = canonicalizer.build_canonical_map(scenes)

    # تطبيق التوحيد
    normalized_scenes = canonicalizer.apply_normalization(scenes)

    # حفظ السجل إذا تم تحديد المسار
    if merge_log_path:
        canonicalizer.export_merge_log(merge_log_path)

    return normalized_scenes, canonical_map, canonicalizer.get_statistics()


# ---------------------------------------------------------
# للتشغيل المستقل (للاختبار)
# ---------------------------------------------------------
if __name__ == "__main__":
    # اختبار بسيط
    print("اختبار وحدة توحيد الكيانات")
    print("=" * 50)

    canonicalizer = EntityCanonicalizer(similarity_threshold=0.85)

    # اختبار حساب التشابه
    test_pairs = [
        ("رأفت", "رأفت الهجان"),
        ("محمد", "محمود"),
        ("أحمد", "أحمد السعيد"),
        ("فاطمة", "فاطمه"),
    ]

    print("\nاختبار حساب التشابه:")
    for name1, name2 in test_pairs:
        similarity = canonicalizer.calculate_similarity(name1, name2)
        print(f"  '{name1}' ↔ '{name2}': {similarity:.2%}")

    # اختبار بناء القاموس من قائمة أسماء
    test_names = [
        "رأفت", "رأفت", "رأفت الهجان", "رأفت",
        "محمد", "محمد", "محمد السيد",
        "أحمد", "أحمد"
    ]

    print("\nاختبار بناء قاموس التطبيع:")
    canonical_map = canonicalizer.build_canonical_map_from_names(test_names)

    for original, canonical in canonical_map.items():
        print(f"  '{original}' → '{canonical}'")

    print("\nالإحصائيات:")
    stats = canonicalizer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
