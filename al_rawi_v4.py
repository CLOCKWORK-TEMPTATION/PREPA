"""
نظام الراوي الإصدار 4.0
وحدات إثراء السياق وفلترة الجودة

الوكيل 3: وحدة إثراء السياق (Context Enrichment)
الوكيل 4: وحدة فلترة الجودة (Quality Filter)
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, List, Dict
from pathlib import Path

# إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('al_rawi_v4.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# نماذج البيانات
# ============================================================

@dataclass
class DialogueTurn:
    """
    وحدة حوار واحدة لشخصية معينة
    """
    scene_id: str
    turn_id: int
    speaker: str
    text: str
    normalized_text: str = ""
    sentiment: str = "unknown"
    sentiment_score: float = 0.0
    original_speaker: str = ""  # للاحتفاظ بالاسم الأصلي بعد التطبيع
    element_ids: List[str] = field(default_factory=list)


@dataclass
class Scene:
    """
    مشهد في السيناريو يحتوي على حوارات وأحداث
    """
    scene_id: str
    scene_number: Optional[int]
    heading: Optional[str]
    location: Optional[str]
    time_of_day: Optional[str]
    int_ext: Optional[str]
    time_period: str = "غير محدد"  # حقل الميتاداتا الزمنية
    actions: List[str] = field(default_factory=list)
    dialogue: List[DialogueTurn] = field(default_factory=list)
    transitions: List[str] = field(default_factory=list)
    element_ids: List[str] = field(default_factory=list)
    full_text: str = ""
    characters: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None


# ============================================================
# دوال مساعدة
# ============================================================

def count_arabic_words(text: str) -> int:
    """
    عد الكلمات العربية في النص

    Args:
        text: النص المراد عد كلماته

    Returns:
        عدد الكلمات
    """
    if not text:
        return 0
    # تنظيف النص وتقسيمه
    words = text.strip().split()
    return len([w for w in words if w.strip()])


def is_transition_line(line: str) -> bool:
    """
    التحقق مما إذا كان السطر يمثل انتقال (transition)

    Args:
        line: السطر المراد فحصه

    Returns:
        True إذا كان السطر انتقال
    """
    transitions = {"قطع", "كات", "CUT", "CUT TO", "FADE OUT", "FADE IN", "DISSOLVE"}
    return line.strip().upper() in transitions or line.strip() in transitions


def is_significant_action(action: str, min_length: int = 10) -> bool:
    """
    التحقق مما إذا كان السطر الوصفي مهماً

    Args:
        action: السطر الوصفي
        min_length: الحد الأدنى لطول السطر المهم

    Returns:
        True إذا كان السطر مهماً
    """
    if not action:
        return False
    clean = action.strip()
    # استبعاد الانتقالات والأسطر القصيرة
    if is_transition_line(clean):
        return False
    if len(clean) < min_length:
        return False
    return True


# ============================================================
# Task 3: وحدة إثراء السياق (Context Enricher)
# ============================================================

class ContextEnricher:
    """
    وحدة إثراء السياق
    مسؤولة عن إضافة السياق الوصفي للحوارات

    المتطلبات: 2.1, 2.2, 2.3, 2.4, 2.5
    """

    def __init__(self, min_action_length: int = 10):
        """
        تهيئة وحدة إثراء السياق

        Args:
            min_action_length: الحد الأدنى لطول السطر الوصفي المهم
        """
        self.min_action_length = min_action_length
        self.enrichment_stats = {
            "scenes_processed": 0,
            "dialogues_enriched": 0,
            "actions_used": 0
        }
        logger.info("تم تهيئة وحدة إثراء السياق")

    def _get_last_significant_action(self, actions: List[str]) -> str:
        """
        استخراج آخر سطر وصفي مهم من قائمة الأفعال

        المتطلب 2.3: استخراج الأسطر الوصفية من قائمة actions

        Args:
            actions: قائمة الأسطر الوصفية

        Returns:
            آخر سطر وصفي مهم، أو سلسلة فارغة إذا لم يوجد
        """
        if not actions:
            return ""

        # البحث من النهاية للعثور على آخر سطر وصفي مهم
        for action in reversed(actions):
            if is_significant_action(action, self.min_action_length):
                self.enrichment_stats["actions_used"] += 1
                logger.debug(f"تم العثور على سطر وصفي مهم: {action[:50]}...")
                return action.strip()

        return ""

    def _build_enriched_scene_setup(
        self,
        scene: Scene,
        last_action: str
    ) -> str:
        """
        بناء وصف المشهد المحسّن مع السياق الوصفي

        المتطلب 2.2: دمج معلومات المكان والزمان مع السياق الوصفي
        المتطلب 2.4: استخدام التنسيق "المكان: X. [سياق: Y]. المتحدث: Z"

        Args:
            scene: كائن المشهد
            last_action: آخر سطر وصفي مهم

        Returns:
            وصف المشهد المحسّن
        """
        parts = []

        # إضافة عنوان المشهد إذا وجد
        if scene.heading:
            parts.append(scene.heading)

        # إضافة معلومات المكان
        if scene.location:
            location_info = f"المكان: {scene.location}"
            if scene.int_ext:
                location_info += f" ({scene.int_ext})"
            parts.append(location_info)

        # إضافة معلومات الزمان
        if scene.time_of_day:
            parts.append(f"الوقت: {scene.time_of_day}")

        # إضافة الفترة الزمنية إذا كانت محددة
        if scene.time_period and scene.time_period != "غير محدد":
            parts.append(f"الفترة: {scene.time_period}")

        # إضافة السياق الوصفي إذا وجد
        if last_action:
            parts.append(f"[سياق: {last_action}]")

        return "\n".join(parts) if parts else ""

    def _get_action_before_turn(
        self,
        actions: List[str],
        turn_index: int,
        dialogue_count: int
    ) -> str:
        """
        استخراج السطر الوصفي الذي يسبق حوار معين

        هذه الدالة تحاول تقدير السطر الوصفي المرتبط بكل حوار
        بناءً على موضعه في التسلسل

        Args:
            actions: قائمة الأسطر الوصفية
            turn_index: فهرس الحوار الحالي
            dialogue_count: العدد الإجمالي للحوارات

        Returns:
            السطر الوصفي المناسب أو سلسلة فارغة
        """
        if not actions or dialogue_count == 0:
            return ""

        # حساب نسبة موضع الحوار في المشهد
        position_ratio = turn_index / dialogue_count

        # تقدير فهرس السطر الوصفي المناسب
        action_index = int(position_ratio * len(actions))
        action_index = min(action_index, len(actions) - 1)

        action = actions[action_index]
        if is_significant_action(action, self.min_action_length):
            return action.strip()

        return ""

    def export_contextual_alpaca(
        self,
        scenes: List[Scene],
        max_context_turns: int = 6
    ) -> List[Dict[str, Any]]:
        """
        تصدير بيانات التدريب بصيغة Alpaca مع سياق محسّن

        المتطلب 2.1: تضمين آخر سطر وصفي مهم قبل الحوار في حقل الإدخال
        المتطلب 2.5: استخدام عنوان المشهد فقط عند عدم وجود سياق وصفي

        Args:
            scenes: قائمة المشاهد
            max_context_turns: الحد الأقصى لعدد الحوارات في السياق

        Returns:
            قائمة بسجلات Alpaca المحسّنة
        """
        data = []

        for scene in scenes:
            self.enrichment_stats["scenes_processed"] += 1
            dialogue = scene.dialogue

            if not dialogue:
                continue

            # استخراج آخر سطر وصفي مهم
            last_action = self._get_last_significant_action(scene.actions)

            # بناء وصف المشهد المحسّن
            scene_setup = self._build_enriched_scene_setup(scene, last_action)

            context_buffer = []

            for i, turn in enumerate(dialogue):
                self.enrichment_stats["dialogues_enriched"] += 1

                # إضافة السياق الوصفي قبل كل حوار
                action_context = self._get_action_before_turn(
                    scene.actions,
                    i,
                    len(dialogue)
                )

                # بناء السياق من الحوارات السابقة
                start = max(0, i - max_context_turns)
                context_turns = context_buffer[start:]
                current_history = "\n".join(context_turns) if context_turns else "بداية الحوار."

                # بناء حقل الإدخال الكامل
                full_input = scene_setup

                if action_context and action_context != last_action:
                    full_input += f"\n[سياق: {action_context}]"

                full_input += f"\n\nسياق الحديث السابق:\n{current_history}"
                full_input += f"\n\nالمتحدث: {turn.speaker}"

                # إنشاء سجل Alpaca
                record = {
                    "instruction": "أكمل الحوار التالي بناءً على السياق المعطى",
                    "input": full_input.strip(),
                    "output": turn.text,
                    "metadata": {
                        "scene_id": scene.scene_id,
                        "turn_id": turn.turn_id,
                        "speaker": turn.speaker,
                        "location": scene.location,
                        "time_of_day": scene.time_of_day,
                        "time_period": scene.time_period,
                        "sentiment": turn.sentiment,
                        "sentiment_score": turn.sentiment_score
                    }
                }

                data.append(record)

                # إضافة الحوار الحالي إلى المخزن المؤقت
                context_buffer.append(f"{turn.speaker}: {turn.text}")

        logger.info(
            f"تم إثراء {self.enrichment_stats['dialogues_enriched']} حوار "
            f"من {self.enrichment_stats['scenes_processed']} مشهد "
            f"باستخدام {self.enrichment_stats['actions_used']} سطر وصفي"
        )

        return data

    def get_enrichment_stats(self) -> Dict[str, int]:
        """
        الحصول على إحصائيات الإثراء

        Returns:
            قاموس بإحصائيات الإثراء
        """
        return self.enrichment_stats.copy()

    def reset_stats(self):
        """إعادة تعيين الإحصائيات"""
        self.enrichment_stats = {
            "scenes_processed": 0,
            "dialogues_enriched": 0,
            "actions_used": 0
        }


# ============================================================
# Task 4: وحدة فلترة الجودة (Quality Filter)
# ============================================================

class QualityFilter:
    """
    وحدة فلترة الجودة
    مسؤولة عن إزالة الحوارات منخفضة الجودة

    المتطلبات: 3.1, 3.2, 3.3, 3.4, 3.5
    """

    def __init__(
        self,
        min_words: int = 3,
        high_sentiment_threshold: float = 0.8
    ):
        """
        تهيئة وحدة فلترة الجودة

        Args:
            min_words: الحد الأدنى لعدد الكلمات للاحتفاظ بالحوار
            high_sentiment_threshold: عتبة المشاعر العالية للاحتفاظ بالحوارات القصيرة
        """
        self.min_words = min_words
        self.sentiment_threshold = high_sentiment_threshold
        self.filter_stats = {
            "total_dialogues": 0,
            "kept_dialogues": 0,
            "filtered_short": 0,
            "kept_emotional": 0
        }
        logger.info(
            f"تم تهيئة وحدة فلترة الجودة: "
            f"الحد الأدنى للكلمات={min_words}, "
            f"عتبة المشاعر={high_sentiment_threshold}"
        )

    def should_keep_turn(self, turn: DialogueTurn) -> bool:
        """
        تحديد ما إذا كان يجب الاحتفاظ بالحوار

        المتطلب 3.1: حذف الحوارات التي عدد كلماتها أقل من 3 كلمات
        المتطلب 3.2: الاحتفاظ بالحوارات القصيرة ذات المشاعر القوية
        المتطلب 3.4: استخدام نموذج تحليل المشاعر الموجود

        Args:
            turn: وحدة الحوار

        Returns:
            True إذا كان الحوار عالي الجودة ويجب الاحتفاظ به
        """
        word_count = count_arabic_words(turn.text)

        # قاعدة 1: الحوارات الطويلة تُحفظ دائماً
        if word_count >= self.min_words:
            return True

        # قاعدة 2: الحوارات القصيرة ذات المشاعر القوية تُحفظ
        if turn.sentiment_score >= self.sentiment_threshold:
            self.filter_stats["kept_emotional"] += 1
            logger.debug(
                f"الاحتفاظ بحوار قصير عاطفي: "
                f"'{turn.text[:30]}...' (درجة المشاعر: {turn.sentiment_score})"
            )
            return True

        # قاعدة 3: إذا لم يكن تحليل المشاعر متوفراً، احتفظ بالحوار
        if turn.sentiment == "unknown":
            logger.warning(
                f"تحليل المشاعر غير متوفر للحوار '{turn.text[:30]}...' - "
                f"الاحتفاظ بالحوار القصير"
            )
            return True

        # خلاف ذلك، تُحذف
        self.filter_stats["filtered_short"] += 1
        logger.debug(f"حذف حوار قصير: '{turn.text[:30]}...' ({word_count} كلمات)")
        return False

    def filter_dialogue_list(
        self,
        dialogues: List[DialogueTurn]
    ) -> List[DialogueTurn]:
        """
        تطبيق الفلترة على قائمة حوارات

        Args:
            dialogues: قائمة الحوارات

        Returns:
            قائمة الحوارات المفلترة
        """
        filtered = []
        for turn in dialogues:
            self.filter_stats["total_dialogues"] += 1
            if self.should_keep_turn(turn):
                self.filter_stats["kept_dialogues"] += 1
                filtered.append(turn)
        return filtered

    def filter_scenes(self, scenes: List[Scene]) -> List[Scene]:
        """
        تطبيق الفلترة على جميع المشاهد

        المتطلب 3.5: تطبيق الفلترة في مرحلة التصدير قبل إنشاء الملفات

        Args:
            scenes: قائمة المشاهد

        Returns:
            قائمة المشاهد مع الحوارات المفلترة
        """
        filtered_scenes = []

        for scene in scenes:
            # تصفية الحوارات
            filtered_dialogue = self.filter_dialogue_list(scene.dialogue)

            # إنشاء نسخة جديدة من المشهد مع الحوارات المفلترة
            filtered_scene = Scene(
                scene_id=scene.scene_id,
                scene_number=scene.scene_number,
                heading=scene.heading,
                location=scene.location,
                time_of_day=scene.time_of_day,
                int_ext=scene.int_ext,
                time_period=scene.time_period,
                actions=scene.actions,
                dialogue=filtered_dialogue,
                transitions=scene.transitions,
                element_ids=scene.element_ids,
                full_text=scene.full_text,
                characters=list(set(t.speaker for t in filtered_dialogue if t.speaker)),
                embedding=scene.embedding,
                embedding_model=scene.embedding_model
            )
            filtered_scenes.append(filtered_scene)

        # تسجيل الإحصائيات
        self._log_filter_stats()

        return filtered_scenes

    def _log_filter_stats(self):
        """
        تسجيل إحصائيات الفلترة في ملف السجل

        المتطلب 3.3: تسجيل عدد الحوارات المحذوفة في ملف السجل
        """
        filtered_count = self.filter_stats["total_dialogues"] - self.filter_stats["kept_dialogues"]
        logger.info(
            f"إحصائيات الفلترة: "
            f"إجمالي الحوارات: {self.filter_stats['total_dialogues']}, "
            f"تم الاحتفاظ بـ: {self.filter_stats['kept_dialogues']}, "
            f"تم حذف: {filtered_count}, "
            f"حوارات عاطفية قصيرة محفوظة: {self.filter_stats['kept_emotional']}"
        )

    def get_filter_stats(self) -> Dict[str, int]:
        """
        الحصول على إحصائيات الفلترة

        Returns:
            قاموس بإحصائيات الفلترة
        """
        stats = self.filter_stats.copy()
        stats["filtered_count"] = stats["total_dialogues"] - stats["kept_dialogues"]
        return stats

    def reset_stats(self):
        """إعادة تعيين الإحصائيات"""
        self.filter_stats = {
            "total_dialogues": 0,
            "kept_dialogues": 0,
            "filtered_short": 0,
            "kept_emotional": 0
        }


# ============================================================
# وحدة التصدير المحسّنة (DatasetExporter)
# ============================================================

class DatasetExporter:
    """
    مصدّر مجموعات البيانات المحسّن
    يدمج إثراء السياق وفلترة الجودة
    """

    def __init__(
        self,
        output_dir: str,
        min_words: int = 3,
        sentiment_threshold: float = 0.8,
        min_action_length: int = 10,
        apply_quality_filter: bool = True,
        apply_context_enrichment: bool = True
    ):
        """
        تهيئة مصدّر مجموعات البيانات

        Args:
            output_dir: مجلد المخرجات
            min_words: الحد الأدنى لعدد الكلمات
            sentiment_threshold: عتبة المشاعر العالية
            min_action_length: الحد الأدنى لطول السطر الوصفي
            apply_quality_filter: تطبيق فلترة الجودة
            apply_context_enrichment: تطبيق إثراء السياق
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.apply_quality_filter = apply_quality_filter
        self.apply_context_enrichment = apply_context_enrichment

        # تهيئة الوحدات
        self.quality_filter = QualityFilter(min_words, sentiment_threshold)
        self.context_enricher = ContextEnricher(min_action_length)

        logger.info(
            f"تم تهيئة مصدّر البيانات: "
            f"الفلترة={'مفعّلة' if apply_quality_filter else 'معطّلة'}, "
            f"الإثراء={'مفعّل' if apply_context_enrichment else 'معطّل'}"
        )

    def export_alpaca_jsonl(
        self,
        scenes: List[Scene],
        filename: str = "alpaca_dataset.jsonl"
    ) -> Path:
        """
        تصدير بيانات التدريب بصيغة Alpaca JSONL

        Args:
            scenes: قائمة المشاهد
            filename: اسم ملف المخرجات

        Returns:
            مسار الملف المُنشأ
        """
        # تطبيق فلترة الجودة إذا كانت مفعّلة
        if self.apply_quality_filter:
            scenes = self.quality_filter.filter_scenes(scenes)

        # تصدير مع إثراء السياق إذا كان مفعّلاً
        if self.apply_context_enrichment:
            data = self.context_enricher.export_contextual_alpaca(scenes)
        else:
            # تصدير بسيط بدون إثراء
            data = self._export_simple_alpaca(scenes)

        # كتابة الملف
        output_path = self.output_dir / filename
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(f"تم تصدير {len(data)} سجل إلى: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"فشل كتابة الملف {output_path}: {e}")
            raise

    def _export_simple_alpaca(self, scenes: List[Scene]) -> List[Dict[str, Any]]:
        """
        تصدير بسيط بدون إثراء السياق

        Args:
            scenes: قائمة المشاهد

        Returns:
            قائمة بسجلات Alpaca البسيطة
        """
        data = []
        for scene in scenes:
            for turn in scene.dialogue:
                record = {
                    "instruction": "أكمل الحوار التالي",
                    "input": f"المتحدث: {turn.speaker}",
                    "output": turn.text,
                    "metadata": {
                        "scene_id": scene.scene_id,
                        "turn_id": turn.turn_id,
                        "speaker": turn.speaker
                    }
                }
                data.append(record)
        return data

    def export_scenes_jsonl(
        self,
        scenes: List[Scene],
        filename: str = "scenes.jsonl"
    ) -> Path:
        """
        تصدير المشاهد بصيغة JSONL

        Args:
            scenes: قائمة المشاهد
            filename: اسم ملف المخرجات

        Returns:
            مسار الملف المُنشأ
        """
        # تطبيق فلترة الجودة إذا كانت مفعّلة
        if self.apply_quality_filter:
            scenes = self.quality_filter.filter_scenes(scenes)

        output_path = self.output_dir / filename
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for scene in scenes:
                    record = {
                        "scene_id": scene.scene_id,
                        "scene_number": scene.scene_number,
                        "heading": scene.heading,
                        "location": scene.location,
                        "time_of_day": scene.time_of_day,
                        "int_ext": scene.int_ext,
                        "time_period": scene.time_period,
                        "characters": scene.characters,
                        "dialogue_count": len(scene.dialogue),
                        "actions_count": len(scene.actions),
                        "full_text": scene.full_text
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(f"تم تصدير {len(scenes)} مشهد إلى: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"فشل كتابة الملف {output_path}: {e}")
            raise

    def get_all_stats(self) -> Dict[str, Any]:
        """
        الحصول على جميع الإحصائيات

        Returns:
            قاموس بجميع الإحصائيات
        """
        return {
            "quality_filter": self.quality_filter.get_filter_stats(),
            "context_enricher": self.context_enricher.get_enrichment_stats()
        }


# ============================================================
# دوال مساعدة للتكامل
# ============================================================

def create_sample_scene() -> Scene:
    """
    إنشاء مشهد نموذجي للاختبار

    Returns:
        كائن Scene نموذجي
    """
    dialogue = [
        DialogueTurn(
            scene_id="S0001",
            turn_id=1,
            speaker="أحمد",
            text="مرحباً يا صديقي، كيف حالك اليوم؟",
            sentiment="positive",
            sentiment_score=0.7
        ),
        DialogueTurn(
            scene_id="S0001",
            turn_id=2,
            speaker="محمد",
            text="بخير",  # حوار قصير للاختبار
            sentiment="neutral",
            sentiment_score=0.3
        ),
        DialogueTurn(
            scene_id="S0001",
            turn_id=3,
            speaker="أحمد",
            text="أخيراً!",  # حوار قصير عاطفي
            sentiment="positive",
            sentiment_score=0.9
        ),
        DialogueTurn(
            scene_id="S0001",
            turn_id=4,
            speaker="محمد",
            text="نعم، لقد كان يوماً طويلاً ومرهقاً في العمل",
            sentiment="negative",
            sentiment_score=0.6
        )
    ]

    return Scene(
        scene_id="S0001",
        scene_number=1,
        heading="مشهد 1 - داخلي - منزل أحمد - نهار",
        location="منزل أحمد",
        time_of_day="نهار",
        int_ext="داخلي",
        time_period="2024",
        actions=[
            "يدخل محمد من الباب الرئيسي",
            "يجلس أحمد على الأريكة ويقرأ كتاباً",
            "ينظر أحمد إلى محمد بابتسامة ترحيب"
        ],
        dialogue=dialogue,
        characters=["أحمد", "محمد"]
    )


def demo():
    """
    تشغيل عرض توضيحي للوحدات
    """
    print("=" * 60)
    print("عرض توضيحي لنظام الراوي الإصدار 4.0")
    print("وحدات إثراء السياق وفلترة الجودة")
    print("=" * 60)

    # إنشاء مشهد نموذجي
    scene = create_sample_scene()
    scenes = [scene]

    print("\n--- المشهد الأصلي ---")
    print(f"عدد الحوارات: {len(scene.dialogue)}")
    for turn in scene.dialogue:
        print(f"  {turn.speaker}: {turn.text}")

    # إنشاء مصدّر البيانات
    exporter = DatasetExporter(
        output_dir="./demo_output",
        min_words=3,
        sentiment_threshold=0.8,
        apply_quality_filter=True,
        apply_context_enrichment=True
    )

    # تصدير البيانات
    alpaca_path = exporter.export_alpaca_jsonl(scenes, "demo_alpaca.jsonl")

    print("\n--- إحصائيات الفلترة ---")
    stats = exporter.get_all_stats()
    filter_stats = stats["quality_filter"]
    print(f"إجمالي الحوارات: {filter_stats['total_dialogues']}")
    print(f"تم الاحتفاظ بـ: {filter_stats['kept_dialogues']}")
    print(f"تم حذف: {filter_stats['filtered_count']}")
    print(f"حوارات عاطفية قصيرة محفوظة: {filter_stats['kept_emotional']}")

    print("\n--- إحصائيات الإثراء ---")
    enricher_stats = stats["context_enricher"]
    print(f"المشاهد المعالجة: {enricher_stats['scenes_processed']}")
    print(f"الحوارات المثراة: {enricher_stats['dialogues_enriched']}")
    print(f"الأسطر الوصفية المستخدمة: {enricher_stats['actions_used']}")

    print(f"\n--- الملف المُصدّر ---")
    print(f"المسار: {alpaca_path}")

    # قراءة وعرض نموذج من البيانات المصدّرة
    print("\n--- نموذج من البيانات المصدّرة ---")
    with open(alpaca_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 2:  # عرض أول سجلين فقط
                break
            record = json.loads(line)
            print(f"\nسجل {i + 1}:")
            print(f"  المدخل: {record['input'][:100]}...")
            print(f"  المخرج: {record['output']}")

    print("\n" + "=" * 60)
    print("انتهى العرض التوضيحي")
    print("=" * 60)


if __name__ == "__main__":
    demo()
