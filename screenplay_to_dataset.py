import os
import re
import json
import math
import mimetypes
import hashlib
import sqlite3
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path

# ----------------------------
# إعداد نظام التسجيل (Logging)
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# ----------------------------
# الاستيراد الآمن للمكتبات الجديدة
# ----------------------------
RAPIDFUZZ_AVAILABLE = False
DIFFLIB_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
    logger.info("✓ مكتبة rapidfuzz متوفرة")
except ImportError:
    logger.warning("⚠ مكتبة rapidfuzz غير متوفرة، سيتم استخدام difflib")
    try:
        import difflib
        DIFFLIB_AVAILABLE = True
        logger.info("✓ مكتبة difflib متوفرة كبديل")
    except ImportError:
        logger.error("✗ لا توجد مكتبات حساب التشابه - سيتم تخطي توحيد الكيانات")

# hypothesis للاختبارات (اختياري)
HYPOTHESIS_AVAILABLE = False
try:
    from hypothesis import given, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    pass

# ----------------------------
# نمط regex لاستخراج السنوات (الميتاداتا الزمنية)
# ----------------------------
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')

# ----------------------------
# دالة مساعدة لعد الكلمات العربية
# ----------------------------
def count_arabic_words(text: str) -> int:
    """عد الكلمات في النص العربي"""
    if not text:
        return 0
    words = text.split()
    return len([w for w in words if w.strip()])

# ----------------------------
# دالة حساب التشابه بين النصوص
# ----------------------------
def calculate_similarity(name1: str, name2: str) -> float:
    """
    حساب نسبة التشابه بين اسمين باستخدام rapidfuzz أو difflib

    Returns:
        نسبة التشابه (0.0 - 1.0)
    """
    if not name1 or not name2:
        return 0.0

    if RAPIDFUZZ_AVAILABLE:
        from rapidfuzz import fuzz
        return fuzz.ratio(name1, name2) / 100.0
    elif DIFFLIB_AVAILABLE:
        import difflib
        return difflib.SequenceMatcher(None, name1, name2).ratio()
    else:
        return 0.0

# ----------------------------
# فئة توحيد الكيانات (EntityCanonicalizer)
# ----------------------------
class EntityCanonicalizer:
    """
    مسؤول عن توحيد أسماء الشخصيات المتشابهة
    يستخدم خوارزمية المسافة الليفنشتاين لحساب التشابه
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: نسبة التشابه المطلوبة للدمج (افتراضي: 85%)
        """
        self.threshold = similarity_threshold
        self.canonical_map: Dict[str, str] = {}
        self.merge_log: List[Dict[str, Any]] = []
        self.stats = {
            "total_names": 0,
            "unique_names_before": 0,
            "unique_names_after": 0,
            "merges_performed": 0
        }

    def _count_occurrences(self, scenes: List['Scene']) -> Dict[str, int]:
        """عد تكرارات كل اسم شخصية"""
        counts: Dict[str, int] = {}
        for scene in scenes:
            for turn in scene.dialogue:
                speaker = turn.speaker.strip()
                if speaker:
                    counts[speaker] = counts.get(speaker, 0) + 1
        return counts

    def build_canonical_map(self, scenes: List['Scene']) -> Dict[str, str]:
        """
        بناء قاموس التطبيع من جميع المشاهد

        Returns:
            قاموس يربط الأسماء المتشابهة بالاسم الكانوني
        """
        if not (RAPIDFUZZ_AVAILABLE or DIFFLIB_AVAILABLE):
            logger.warning("لا تتوفر مكتبات حساب التشابه - تخطي توحيد الكيانات")
            return {}

        occurrences = self._count_occurrences(scenes)
        names = list(occurrences.keys())

        self.stats["total_names"] = sum(occurrences.values())
        self.stats["unique_names_before"] = len(names)

        if len(names) < 2:
            self.stats["unique_names_after"] = len(names)
            return {}

        # مجموعات الأسماء المتشابهة
        processed = set()
        groups: List[List[str]] = []

        for i, name1 in enumerate(names):
            if name1 in processed:
                continue

            group = [name1]
            processed.add(name1)

            for j, name2 in enumerate(names[i+1:], i+1):
                if name2 in processed:
                    continue

                similarity = calculate_similarity(name1, name2)
                if similarity >= self.threshold:
                    group.append(name2)
                    processed.add(name2)

            if len(group) > 1:
                groups.append(group)

        # بناء قاموس التطبيع
        for group in groups:
            # اختيار الاسم الكانوني: الأكثر تكراراً، ثم الأطول
            canonical = max(group, key=lambda x: (occurrences.get(x, 0), len(x)))

            for name in group:
                if name != canonical:
                    self.canonical_map[name] = canonical
                    self.merge_log.append({
                        "original": name,
                        "canonical": canonical,
                        "similarity": calculate_similarity(name, canonical),
                        "original_count": occurrences.get(name, 0),
                        "canonical_count": occurrences.get(canonical, 0)
                    })
                    self.stats["merges_performed"] += 1

        self.stats["unique_names_after"] = self.stats["unique_names_before"] - self.stats["merges_performed"]

        logger.info(f"✓ توحيد الكيانات: {self.stats['merges_performed']} عملية دمج")
        return self.canonical_map

    def normalize_character_name(self, name: str) -> str:
        """
        تطبيع اسم شخصية واحدة

        Returns:
            الاسم الكانوني
        """
        return self.canonical_map.get(name.strip(), name.strip())

    def apply_normalization(self, scenes: List['Scene']) -> List['Scene']:
        """
        تطبيق التطبيع على جميع الحوارات في المشاهد
        """
        if not self.canonical_map:
            return scenes

        for scene in scenes:
            for turn in scene.dialogue:
                original = turn.speaker
                turn.speaker = self.normalize_character_name(original)

            # تحديث قائمة الشخصيات
            scene.characters = list(set(
                self.normalize_character_name(c) for c in scene.characters
            ))

        return scenes

    def export_merge_log(self, output_path: str):
        """حفظ سجل عمليات الدمج"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "stats": self.stats,
                    "merge_log": self.merge_log,
                    "canonical_map": self.canonical_map
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ تم حفظ سجل الدمج في: {output_path}")
        except Exception as e:
            logger.error(f"✗ فشل حفظ سجل الدمج: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التوحيد"""
        return self.stats.copy()


# ----------------------------
# فئة فلترة الجودة (QualityFilter)
# ----------------------------
class QualityFilter:
    """
    فلترة الحوارات منخفضة الجودة
    """

    def __init__(self, min_words: int = 3, high_sentiment_threshold: float = 0.8):
        """
        Args:
            min_words: الحد الأدنى لعدد الكلمات (افتراضي: 3)
            high_sentiment_threshold: عتبة المشاعر العالية (افتراضي: 0.8)
        """
        self.min_words = min_words
        self.sentiment_threshold = high_sentiment_threshold
        self.filtered_count = 0
        self.kept_count = 0
        self.kept_by_sentiment = 0
        self.filter_log: List[Dict[str, Any]] = []

    def should_keep_turn(self, turn: 'DialogueTurn') -> bool:
        """
        تحديد ما إذا كان يجب الاحتفاظ بالحوار

        Returns:
            True إذا كان الحوار عالي الجودة
        """
        word_count = count_arabic_words(turn.text)

        # قاعدة 1: الحوارات الطويلة تُحفظ دائماً
        if word_count >= self.min_words:
            self.kept_count += 1
            return True

        # قاعدة 2: الحوارات القصيرة ذات المشاعر القوية تُحفظ
        sentiment_score = getattr(turn, 'sentiment_score', 0.0)
        if sentiment_score >= self.sentiment_threshold:
            self.kept_by_sentiment += 1
            self.kept_count += 1
            logger.debug(f"احتفاظ بحوار قصير بسبب المشاعر القوية: {turn.text[:30]}...")
            return True

        # تسجيل الحوار المفلتر
        self.filter_log.append({
            "scene_id": turn.scene_id,
            "turn_id": turn.turn_id,
            "speaker": turn.speaker,
            "text": turn.text,
            "word_count": word_count,
            "reason": "short_dialogue"
        })
        self.filtered_count += 1
        return False

    def filter_scenes(self, scenes: List['Scene']) -> List['Scene']:
        """
        تطبيق الفلترة على جميع المشاهد
        """
        filtered_scenes = []

        for scene in scenes:
            filtered_dialogue = [
                turn for turn in scene.dialogue
                if self.should_keep_turn(turn)
            ]

            # إنشاء نسخة جديدة من المشهد مع الحوارات المفلترة
            new_scene = Scene(
                scene_id=scene.scene_id,
                scene_number=scene.scene_number,
                heading=scene.heading,
                location=scene.location,
                time_of_day=scene.time_of_day,
                int_ext=scene.int_ext,
                time_period=getattr(scene, 'time_period', 'غير محدد'),
                actions=scene.actions.copy(),
                dialogue=filtered_dialogue,
                transitions=scene.transitions.copy(),
                element_ids=scene.element_ids.copy(),
                full_text=scene.full_text,
                characters=list(set(t.speaker for t in filtered_dialogue if t.speaker)),
                embedding=scene.embedding,
                embedding_model=scene.embedding_model
            )
            filtered_scenes.append(new_scene)

        logger.info(f"✓ فلترة الجودة: تمت إزالة {self.filtered_count} حوار من أصل {self.filtered_count + self.kept_count}")
        return filtered_scenes

    def get_stats(self) -> Dict[str, int]:
        """الحصول على إحصائيات الفلترة"""
        return {
            "filtered_count": self.filtered_count,
            "kept_count": self.kept_count,
            "kept_by_sentiment": self.kept_by_sentiment,
            "total_processed": self.filtered_count + self.kept_count
        }

    def export_filter_log(self, output_path: str):
        """حفظ سجل الفلترة"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "stats": self.get_stats(),
                    "filtered_dialogues": self.filter_log
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ تم حفظ سجل الفلترة في: {output_path}")
        except Exception as e:
            logger.error(f"✗ فشل حفظ سجل الفلترة: {e}")


# ----------------------------
# فئة إثراء السياق (ContextEnricher)
# ----------------------------
class ContextEnricher:
    """
    إثراء السياق للحوارات قبل التصدير
    """

    @staticmethod
    def get_last_significant_action(actions: List[str], min_length: int = 10) -> str:
        """
        استخراج آخر سطر وصفي مهم (أكثر من min_length حرفاً)
        """
        for action in reversed(actions):
            cleaned = action.strip()
            if len(cleaned) >= min_length and cleaned not in TRANSITIONS:
                return cleaned
        return ""

    @staticmethod
    def build_enriched_scene_setup(scene: 'Scene', last_action: str = "") -> str:
        """
        بناء وصف المشهد مع السياق الوصفي
        التنسيق: "المكان: X. [سياق: Y]. المتحدث: Z"
        """
        parts = []

        # المكان
        if scene.location:
            parts.append(f"المكان: {scene.location}")
        elif scene.heading:
            parts.append(f"المشهد: {scene.heading}")

        # الوقت
        if scene.time_of_day:
            parts.append(f"الوقت: {scene.time_of_day}")

        # داخلي/خارجي
        if scene.int_ext:
            parts.append(f"النوع: {scene.int_ext}")

        # الفترة الزمنية
        time_period = getattr(scene, 'time_period', None)
        if time_period and time_period != "غير محدد":
            parts.append(f"السنة: {time_period}")

        setup = ". ".join(parts)

        # إضافة السياق الوصفي
        if last_action:
            setup += f"\n[سياق: {last_action}]"

        return setup

    @staticmethod
    def format_contextual_input(scene: 'Scene', turn: 'DialogueTurn',
                                context_buffer: List[str], last_action: str = "") -> str:
        """
        تنسيق الإدخال مع السياق الكامل
        """
        scene_setup = ContextEnricher.build_enriched_scene_setup(scene, last_action)

        current_history = "\n".join(context_buffer) if context_buffer else "بداية الحوار."

        full_input = f"{scene_setup}\n\nسياق الحديث السابق:\n{current_history}\n\nالمتحدث: {turn.speaker}"

        return full_input


# ----------------------------
# دالة استخراج الفترة الزمنية
# ----------------------------
def extract_time_period(text: str, last_known: str = "غير محدد") -> Tuple[str, str]:
    """
    استخراج الفترة الزمنية من نص المشهد

    Args:
        text: نص عنوان المشهد أو محتواه
        last_known: آخر سنة معروفة للوراثة

    Returns:
        tuple: (السنة المستخرجة, آخر سنة معروفة للمشهد التالي)
    """
    try:
        match = YEAR_PATTERN.search(text or "")
        if match:
            year = match.group(0)
            return year, year
        return last_known, last_known
    except Exception as e:
        logger.error(f"فشل استخراج الفترة الزمنية: {e}")
        return "غير محدد", last_known


# ----------------------------
# استيراد وحدة معالجة الأخطاء والتسجيل (المتطلب 6)
# ----------------------------
try:
    from error_handling import (
        ArabicLogger,
        ErrorHandler,
        DataValidator,
        SafeWriter,
        StatisticsCollector,
        LogLevel,
        setup_error_handling,
        safe_import,
        get_logger,
        set_logger,
        ARABIC_ERROR_MESSAGES,
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False
    print("تحذير: وحدة معالجة الأخطاء غير متوفرة. سيتم استخدام الوضع الأساسي.")

# ----------------------------
# 1) Open-source Unstructured (local partition)
# ----------------------------
def local_partition(input_path: str) -> list[dict[str, Any]]:
    """
    Partitions a file locally using the open-source `unstructured` library.
    Falls back to reading an existing elements JSON if the input already is a JSON list.
    """
    if input_path.lower().endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
            return data
        raise ValueError("JSON input must be a list of Unstructured elements dicts.")

    try:
        from unstructured.partition.auto import partition
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: unstructured. Install with: pip install unstructured"
        ) from e

    elements = partition(filename=input_path)
    out: list[dict[str, Any]] = []
    for el in elements:
        # Most Element objects support to_dict(); if not, fall back to minimal shape
        if hasattr(el, "to_dict"):
            d = el.to_dict()
        else:
            d = {"type": el.__class__.__name__, "text": getattr(el, "text", "")}
        # Normalize keys we rely on
        if "element_id" not in d:
            d["element_id"] = getattr(el, "id", None) or hashlib.sha1(
                (d.get("type", "") + "::" + d.get("text", "")).encode("utf-8", errors="ignore")
            ).hexdigest()
        out.append(d)
    return out


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown_to_elements(markdown_text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, raw_line in enumerate((markdown_text or "").splitlines(), 1):
        line = (raw_line or "").strip()
        if not line:
            continue
        eid = hashlib.sha1(f"md:{i}:{line}".encode("utf-8", errors="ignore")).hexdigest()
        out.append({"type": "Text", "text": line, "element_id": eid, "source": "docling_markdown"})
    return out


def _extract_metadata_from_text(text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()][:40]
    for ln in lines:
        if ln.startswith("#") and "title" not in meta:
            meta["title"] = ln.lstrip("#").strip()
        if ("تأليف" in ln or "المؤلف" in ln) and "author" not in meta:
            meta["author"] = ln.replace("تأليف", "").replace("المؤلف", "").strip()
        if ("إخراج" in ln or "المخرج" in ln) and "director" not in meta:
            meta["director"] = ln.replace("إخراج", "").replace("المخرج", "").strip()
        year_match = re.search(r"\b(19|20)\d{2}\b", ln)
        if year_match and "year" not in meta:
            meta["year"] = year_match.group()
    return meta


def _docling_extract(input_path: str, ocr_languages: Optional[list[str]] = None, num_threads: int = 4) -> dict[str, Any]:
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    except ImportError as e:
        raise RuntimeError("Missing dependency: docling. Install with: pip install docling") from e

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(
        lang=ocr_languages or ["ar", "en"],
        force_full_page_ocr=False,
    )
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=int(num_threads), device=AcceleratorDevice.AUTO)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    result = converter.convert(str(input_path))
    doc = result.document

    md = ""
    try:
        md = doc.export_to_markdown() or ""
    except Exception:
        md = ""

    doctags = ""
    try:
        doctags = doc.export_to_doctags() or ""
    except Exception:
        doctags = ""

    raw = None
    try:
        raw = doc.export_to_dict()
    except Exception:
        raw = None

    return {"markdown": md, "doctags": doctags, "raw": raw}


def elements_from_input(
    input_path: str,
    extractor: str,
    out_dir: str,
    save_docling_artifacts: bool,
    docling_ocr_languages: Optional[list[str]] = None,
    docling_threads: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extractor = (extractor or "auto").strip().lower()
    input_lower = (input_path or "").lower()
    meta: dict[str, Any] = {
        "input_path": input_path,
    }

    if os.path.isfile(input_path):
        try:
            meta["input_sha1"] = _file_sha1(input_path)
        except Exception:
            meta["input_sha1"] = None

    if extractor == "auto":
        extractor = "docling" if input_lower.endswith(".pdf") else "unstructured"

    if extractor == "docling":
        artifacts = _docling_extract(
            input_path=input_path,
            ocr_languages=docling_ocr_languages,
            num_threads=docling_threads,
        )
        markdown_text = artifacts.get("markdown") or ""

        meta["extractor"] = "docling"
        meta["metadata"] = _extract_metadata_from_text(markdown_text)

        if save_docling_artifacts:
            os.makedirs(out_dir, exist_ok=True)
            md_path = os.path.join(out_dir, "docling.markdown.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            meta["docling_markdown_path"] = md_path

            doctags = artifacts.get("doctags") or ""
            if doctags:
                dt_path = os.path.join(out_dir, "docling.doctags.txt")
                with open(dt_path, "w", encoding="utf-8") as f:
                    f.write(doctags)
                meta["docling_doctags_path"] = dt_path

            raw = artifacts.get("raw")
            if raw is not None:
                raw_path = os.path.join(out_dir, "docling.raw.json")
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(raw, f, ensure_ascii=False, indent=2)
                meta["docling_raw_path"] = raw_path

        return markdown_to_elements(markdown_text), meta

    meta["extractor"] = "unstructured"
    return local_partition(input_path), meta

# ----------------------------
# 2) Screenplay structuring (scenes/dialogue/actions)
# ----------------------------
SCENE_RE = re.compile(r"^\s*(?:المشهد|مشهد)\s*(\d+)\b", re.IGNORECASE)
TIME_HINTS = ["ليل", "نهار", "صباح", "مساء", "فجر", "غروب"]
INT_EXT_HINTS = ["داخلي", "خارجي"]
SPEAKER_RE = re.compile(r"^\s*([^\n:]{1,40})\s*:\s*$")
SPEAKER_INLINE_RE = re.compile(r"^\s*([^\n:]{1,40})\s*:\s*(.+?)\s*$")
TRANSITIONS = {"قطع", "كات", "CUT", "CUT TO", "FADE OUT", "FADE IN"}
STAGE_KEYWORDS = [
    "يجلس",
    "يقف",
    "يدخل",
    "يخرج",
    "ينظر",
    "يتحدث",
    "تجلس",
    "تقف",
    "تدخل",
    "تخرج",
    "تنظر",
    "تتحدث",
    "ينهض",
    "تنهض",
    "يمشي",
    "تمشي",
    "بينما",
    "فجأة",
]

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _parse_scene_heading(line: str) -> Optional[dict[str, Any]]:
    clean = re.sub(r"^[#>\-\*\s]+", "", (line or "")).strip()
    m = SCENE_RE.match(clean)
    if not m:
        return None
    num = int(m.group(1))
    time_of_day = next((t for t in TIME_HINTS if t in clean), None)
    int_ext = next((ie for ie in INT_EXT_HINTS if ie in clean), None)
    return {"scene_number": num, "heading": clean.strip(), "time_of_day": time_of_day, "int_ext": int_ext}


def _looks_like_character_name(name: str) -> bool:
    n = _norm_ws(name)
    if not n:
        return False
    if len(n) > 40:
        return False
    if len(n.split()) > 3:
        return False
    if any(k in n for k in STAGE_KEYWORDS):
        return False
    return True


def _speaker_inline(line: str) -> Optional[tuple[str, str]]:
    m = SPEAKER_INLINE_RE.match(line)
    if not m:
        return None
    name = m.group(1).strip()
    rest = m.group(2).strip()
    if not _looks_like_character_name(name):
        return None
    return name, rest

def _is_speaker(line: str) -> bool:
    if SPEAKER_RE.match(line):
        return _looks_like_character_name(_speaker_name(line))
    return _speaker_inline(line) is not None

def _speaker_name(line: str) -> str:
    m = SPEAKER_RE.match(line)
    return m.group(1).strip() if m else ""

@dataclass
class DialogueTurn:
    scene_id: str
    turn_id: int
    speaker: str
    text: str
    element_ids: list[str] = field(default_factory=list)

@dataclass
class Scene:
    scene_id: str
    scene_number: Optional[int]
    heading: Optional[str]
    location: Optional[str]
    time_of_day: Optional[str]
    int_ext: Optional[str]
    time_period: str = "غير محدد"  # حقل جديد للفترة الزمنية (المتطلب 4)
    time_period: str = "غير محدد"  # حقل جديد للفترة الزمنية
    actions: list[str] = field(default_factory=list)
    dialogue: list[DialogueTurn] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)
    element_ids: list[str] = field(default_factory=list)
    full_text: str = ""
    characters: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None


# ----------------------------
# 2.5) وحدة استخراج الميتاداتا الزمنية (Temporal Metadata Extractor)
# المتطلب 4: استخراج الميتاداتا الزمنية
# ----------------------------
class TemporalMetadataExtractor:
    """
    مسؤول عن استخراج الفترات الزمنية من عناوين المشاهد ومحتواها.
    يبحث عن السنوات باستخدام نمط regex ويدعم وراثة الفترة الزمنية بين المشاهد.
    """

    # نمط regex للبحث عن السنوات (1900-2099)
    YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')

    # مؤشرات زمنية إضافية في النص العربي
    TEMPORAL_INDICATORS = [
        "فلاش باك", "فلاشباك", "ذكريات", "الماضي",
        "سنوات مضت", "عام", "سنة", "قبل"
    ]

    def __init__(self):
        """تهيئة المستخرج مع قيمة افتراضية للسنة الأخيرة المعروفة"""
        self.last_known_year: str = "غير محدد"
        self.extraction_log: list[dict[str, Any]] = []

    def extract_time_period(self, text: str, search_content: bool = True) -> str:
        """
        استخراج الفترة الزمنية من النص.

        Args:
            text: نص عنوان المشهد أو محتواه
            search_content: البحث في محتوى المشهد وليس فقط العنوان

        Returns:
            السنة إن وُجدت، أو آخر سنة معروفة، أو "غير محدد"
        """
        try:
            if not text:
                return self.last_known_year

            # البحث عن السنوات في النص
            match = self.YEAR_PATTERN.search(text)
            if match:
                year = match.group(0)
                self.last_known_year = year
                self._log_extraction(text, year, "found")
                return year

            # إذا لم نجد سنة، نرث من المشهد السابق
            return self.last_known_year

        except Exception as e:
            self._log_extraction(text, "غير محدد", f"error: {str(e)}")
            return "غير محدد"

    def extract_from_heading_and_content(self, heading: str, content: str) -> str:
        """
        استخراج الفترة الزمنية من عنوان المشهد ومحتواه.
        يبحث أولاً في العنوان، ثم في المحتوى إذا لم يجد.

        Args:
            heading: عنوان المشهد
            content: محتوى المشهد (الأحداث والحوارات)

        Returns:
            السنة المستخرجة أو القيمة الافتراضية
        """
        # البحث في العنوان أولاً
        if heading:
            match = self.YEAR_PATTERN.search(heading)
            if match:
                year = match.group(0)
                self.last_known_year = year
                self._log_extraction(heading, year, "found_in_heading")
                return year

        # البحث في المحتوى
        if content:
            match = self.YEAR_PATTERN.search(content)
            if match:
                year = match.group(0)
                self.last_known_year = year
                self._log_extraction(content[:100], year, "found_in_content")
                return year

        # وراثة من المشهد السابق
        return self.last_known_year

    def reset(self):
        """إعادة تعيين حالة المستخرج"""
        self.last_known_year = "غير محدد"
        self.extraction_log = []

    def apply_to_scenes(self, scenes: list["Scene"]) -> list["Scene"]:
        """
        تطبيق استخراج الفترة الزمنية على قائمة المشاهد.

        Args:
            scenes: قائمة المشاهد المراد معالجتها

        Returns:
            قائمة المشاهد مع حقل time_period محدث
        """
        self.reset()

        for scene in scenes:
            # تجميع محتوى المشهد للبحث فيه
            content_parts = []
            if scene.actions:
                content_parts.extend(scene.actions)
            if scene.dialogue:
                for turn in scene.dialogue:
                    content_parts.append(turn.text)

            content = " ".join(content_parts)

            # استخراج الفترة الزمنية
            time_period = self.extract_from_heading_and_content(
                scene.heading or "",
                content
            )

            # تحديث المشهد
            scene.time_period = time_period

        return scenes

    def _log_extraction(self, text: str, result: str, status: str):
        """تسجيل عملية الاستخراج"""
        self.extraction_log.append({
            "text_preview": text[:50] if text else "",
            "result": result,
            "status": status
        })

    def get_extraction_stats(self) -> dict[str, Any]:
        """الحصول على إحصائيات الاستخراج"""
        found_count = sum(1 for log in self.extraction_log if "found" in log.get("status", ""))
        error_count = sum(1 for log in self.extraction_log if "error" in log.get("status", ""))

        return {
            "total_extractions": len(self.extraction_log),
            "found_years": found_count,
            "errors": error_count,
            "inherited": len(self.extraction_log) - found_count - error_count
        }


def elements_to_scenes(elements: list[dict[str, Any]]) -> list[Scene]:
    scenes: list[Scene] = []
    current: Optional[Scene] = None

    current_speaker: Optional[str] = None
    current_turn_text: list[str] = []
    current_turn_eids: list[str] = []
    turn_counter = 0
    pending_location = False
    last_known_year = "غير محدد"  # لوراثة الفترة الزمنية بين المشاهد

    def flush_turn():
        nonlocal current_speaker, current_turn_text, current_turn_eids, turn_counter, current
        if current and current_speaker and any(t.strip() for t in current_turn_text):
            turn_counter += 1
            text = "\n".join([t for t in current_turn_text if t.strip()])
            current.dialogue.append(
                DialogueTurn(
                    scene_id=current.scene_id,
                    turn_id=turn_counter,
                    speaker=current_speaker,
                    text=text,
                    element_ids=current_turn_eids.copy(),
                )
            )
        current_speaker = None
        current_turn_text = []
        current_turn_eids = []

    def finalize_scene(scene: Scene):
        chars: list[str] = []
        for dt in scene.dialogue:
            if dt.speaker and dt.speaker not in chars:
                chars.append(dt.speaker)
        scene.characters = chars

        parts: list[str] = []
        if scene.heading:
            parts.append(scene.heading)
        if scene.location:
            parts.append(scene.location)
        parts.extend(scene.actions)
        for dt in scene.dialogue:
            parts.append(f"{dt.speaker}: {dt.text}")
        parts.extend(scene.transitions)
        scene.full_text = "\n".join([p for p in parts if p.strip()]).strip()

        # استخراج الفترة الزمنية من محتوى المشهد إذا لم تكن موجودة
        if scene.time_period == "غير محدد":
            period, _ = extract_time_period(scene.full_text, scene.time_period)
            scene.time_period = period

    def start_scene(meta: dict[str, Any]):
        nonlocal current, turn_counter, pending_location, last_known_year
        if current:
            flush_turn()
            finalize_scene(current)
            scenes.append(current)

        # استخراج الفترة الزمنية من عنوان المشهد
        heading = meta.get("heading") or ""
        time_period, last_known_year = extract_time_period(heading, last_known_year)

        sid = f"S{meta.get('scene_number', len(scenes) + 1):04d}"
        current = Scene(
            scene_id=sid,
            scene_number=meta.get("scene_number"),
            heading=meta.get("heading"),
            location=None,
            time_of_day=meta.get("time_of_day"),
            int_ext=meta.get("int_ext"),
            time_period=time_period,  # إضافة الفترة الزمنية
        )
        turn_counter = 0
        pending_location = True

    found_scene = any(_parse_scene_heading(_norm_ws(el.get("text", ""))) for el in elements)
    if not found_scene:
        start_scene({"scene_number": 1, "heading": None, "time_of_day": None, "int_ext": None})

    for el in elements:
        text = el.get("text", "") or ""
        line = text.strip()
        if not line:
            continue

        eid = el.get("element_id") or el.get("id") or el.get("elementId")
        meta = _parse_scene_heading(line)

        # content before first scene header: ignore
        if current is None and not meta:
            continue

        if meta:
            start_scene(meta)
            if eid:
                current.element_ids.append(str(eid))
            continue

        assert current is not None
        if eid:
            current.element_ids.append(str(eid))

        # Location heuristic: first short non-speaker line after scene header
        if pending_location and (not _is_speaker(line)) and (line not in TRANSITIONS):
            if len(line) <= 90:
                current.location = line
                pending_location = False
                continue
            # if first thing is a long action, stop waiting for location
            pending_location = False

        if line in TRANSITIONS:
            flush_turn()
            current.transitions.append(line)
            continue

        inline = _speaker_inline(line)
        if inline is not None:
            flush_turn()
            current_speaker = inline[0]
            current_turn_text = [inline[1]] if inline[1] else []
            current_turn_eids = [str(eid)] if eid else []
            continue

        if _is_speaker(line):
            flush_turn()
            current_speaker = _speaker_name(line)
            current_turn_eids = [str(eid)] if eid else []
            continue

        if current_speaker:
            current_turn_text.append(line)
            if eid:
                current_turn_eids.append(str(eid))
            continue

        # Otherwise action/narrative
        current.actions.append(line)

    if current:
        flush_turn()
        finalize_scene(current)
        scenes.append(current)

    return scenes

# ----------------------------
# 3) API layer (Unstructured On-Demand Jobs) to embed scenes
# ----------------------------
def _looks_like_vector(x: Any) -> bool:
    return (
        isinstance(x, list)
        and len(x) >= 8
        and all(isinstance(v, (int, float)) for v in x[:8])
    )

def _collect_vectors(obj: Any, out: list[list[float]]):
    if _looks_like_vector(obj):
        out.append([float(v) for v in obj])
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            # common keys
            if k.lower() in {"embedding", "embeddings", "vector", "vectors"} and _looks_like_vector(v):
                out.append([float(t) for t in v])
            else:
                _collect_vectors(v, out)
    elif isinstance(obj, list):
        for it in obj:
            _collect_vectors(it, out)

def _avg_vectors(vectors: list[list[float]]) -> Optional[list[float]]:
    if not vectors:
        return None
    dim = min(len(v) for v in vectors)
    if dim == 0:
        return None
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    n = float(len(vectors))
    return [x / n for x in acc]

def embed_scenes_via_on_demand_jobs(
    scenes: list[Scene],
    api_key: str,
    work_dir: str,
    batch_size: int = 10,
    embedder_subtype: str = "bedrock",
    embedder_model: str = "cohere.embed-multilingual-v3",
) -> None:
    """
    Splits scenes into batches (<=10 files per job) and runs a custom on-demand job:
    partition -> chunk_by_character -> embed.

    Writes per-batch inputs into work_dir/input and reads outputs from work_dir/output.
    """
    from unstructured_client import UnstructuredClient
    from unstructured_client.models.operations import CreateJobRequest, DownloadJobOutputRequest
    from unstructured_client.models.shared import BodyCreateJob, InputFiles

    os.makedirs(work_dir, exist_ok=True)
    input_root = os.path.join(work_dir, "input")
    output_root = os.path.join(work_dir, "output")
    os.makedirs(input_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    def run_job(client: UnstructuredClient, input_dir: str, job_nodes: list[dict[str, Any]]) -> dict[str, Any]:
        files: list[InputFiles] = []
        for filename in sorted(os.listdir(input_dir)):
            full_path = os.path.join(input_dir, filename)
            if not os.path.isfile(full_path):
                continue
            ctype = mimetypes.guess_type(full_path)[0] or "text/plain"
            files.append(
                InputFiles(
                    content=open(full_path, "rb"),
                    file_name=filename,
                    content_type=ctype,
                )
            )

        request_data = json.dumps({"job_nodes": job_nodes})
        resp = client.jobs.create_job(
            request=CreateJobRequest(
                body_create_job=BodyCreateJob(
                    request_data=request_data,
                    input_files=files,
                )
            )
        )
        job_id = resp.job_information.id
        input_file_ids = resp.job_information.input_file_ids

        # poll
        while True:
            job = client.jobs.get_job(request={"job_id": job_id}).job_information
            if job.status in {"SCHEDULED", "IN_PROGRESS"}:
                # lightweight sleep without importing time at top
                import time
                time.sleep(3)
                continue
            if job.status != "COMPLETED":
                raise RuntimeError(f"Unstructured job failed with status={job.status}")
            break

        # download
        outputs: dict[str, Any] = {}
        for fid in input_file_ids:
            out_resp = client.jobs.download_job_output(
                request=DownloadJobOutputRequest(job_id=job_id, file_id=fid)
            )
            outputs[fid] = out_resp.any
            with open(os.path.join(output_root, f"{fid}.json"), "w", encoding="utf-8") as f:
                json.dump(out_resp.any, f, ensure_ascii=False, indent=2)
        return outputs

    # Custom workflow nodes (per docs: Partitioner/Chunker/Embedder nodes)
    job_nodes = [
        {
            "name": "partition",
            "type": "partition",
            "subtype": "unstructured_api",
            "settings": {
                "strategy": "fast",
                "ocr_languages": ["ara"],
                # إن سببت هذه المفاتيح خطأ عندك، احذفها فقط:
                "unique_element_ids": True,
                "output_format": "json",
            },
        },
        {
            "name": "chunk",
            "type": "chunk",
            "subtype": "chunk_by_character",
            "settings": {
                "max_characters": 3500,
                "new_after_n_chars": 3200,
                "overlap": 200,
            },
        },
        {
            "name": "embed",
            "type": "embed",
            "subtype": embedder_subtype,
            "settings": {
                "model_name": embedder_model,
            },
        },
    ]

    # Map scene -> temp file name, then scene embedding from vectors found in output
    with UnstructuredClient(api_key_auth=api_key) as client:
        for b in range(0, len(scenes), batch_size):
            batch = scenes[b : b + batch_size]

            # create isolated input folder for the batch
            batch_in = os.path.join(input_root, f"batch_{b//batch_size:04d}")
            os.makedirs(batch_in, exist_ok=True)

            scene_file_map: dict[str, Scene] = {}
            for sc in batch:
                fname = f"{sc.scene_id}.txt"
                fpath = os.path.join(batch_in, fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(sc.full_text.strip() + "\n")
                scene_file_map[fname] = sc

            outputs = run_job(client, batch_in, job_nodes)

            # outputs keyed by file_id, so we infer by scanning downloaded json files
            # and using the stored file_id json in output_root. We also keep a best-effort direct parse.
            for fid, obj in outputs.items():
                vectors: list[list[float]] = []
                _collect_vectors(obj, vectors)
                emb = _avg_vectors(vectors)
                # We do not know which input filename maps to fid in all cases,
                # so we also try to match by presence of scene_id text inside output.
                # Best-effort: if exactly one of this batch's scene_ids appears, assign.
                assigned = False
                if isinstance(obj, (list, dict)):
                    blob = json.dumps(obj, ensure_ascii=False)
                    hits = [sc for sc in batch if sc.scene_id in blob]
                    if len(hits) == 1:
                        hits[0].embedding = emb
                        hits[0].embedding_model = f"{embedder_subtype}:{embedder_model}"
                        assigned = True

                # Fallback: assign by order (only if safe)
                if (not assigned) and emb is not None and len(batch) == 1:
                    batch[0].embedding = emb
                    batch[0].embedding_model = f"{embedder_subtype}:{embedder_model}"

# ----------------------------
# 4) Dataset writers
# ----------------------------
def write_jsonl(path: str, rows: list[dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_interactions_index(scenes: list[Scene]) -> list[dict[str, Any]]:
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for sc in scenes:
        chars = [c for c in sc.characters if c]
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                a, b = sorted((chars[i], chars[j]))
                key = (a, b)
                if key not in edges:
                    edges[key] = {
                        "character_a": a,
                        "character_b": b,
                        "scenes": [],
                        "co_occurrence_scenes": 0,
                        "turn_pairs": 0,
                    }
                if sc.scene_id not in edges[key]["scenes"]:
                    edges[key]["scenes"].append(sc.scene_id)
                    edges[key]["co_occurrence_scenes"] += 1

        turns = sc.dialogue
        for i in range(1, len(turns)):
            a0 = turns[i - 1].speaker
            b0 = turns[i].speaker
            if not a0 or not b0 or a0 == b0:
                continue
            a, b = sorted((a0, b0))
            key = (a, b)
            if key not in edges:
                edges[key] = {
                    "character_a": a,
                    "character_b": b,
                    "scenes": [],
                    "co_occurrence_scenes": 0,
                    "turn_pairs": 0,
                }
            edges[key]["turn_pairs"] += 1
    return list(edges.values())


def make_speaker_id_pairs(scenes: list[Scene], max_context_turns: int = 6) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sc in scenes:
        turns = sc.dialogue
        for i, t in enumerate(turns):
            if not t.text.strip() or not t.speaker.strip():
                continue
            start = max(0, i - max_context_turns)
            ctx = turns[start:i]
            prompt_lines: list[str] = []
            if sc.heading:
                prompt_lines.append(sc.heading)
            if sc.location:
                prompt_lines.append(sc.location)
            if ctx:
                prompt_lines.append("الحوار السابق:")
                for ct in ctx:
                    prompt_lines.append(f"{ct.speaker}: {ct.text}")
            prompt_lines.append("النص:")
            prompt_lines.append(t.text)
            prompt_lines.append("من المتحدث؟")
            rows.append(
                {
                    "scene_id": sc.scene_id,
                    "turn_id": t.turn_id,
                    "prompt": "\n".join(prompt_lines).strip(),
                    "target": t.speaker,
                }
            )
    return rows


def write_sqlite_db(
    db_path: str,
    scenes_rows: list[dict[str, Any]],
    dialogue_rows: list[dict[str, Any]],
    characters_rows: list[dict[str, Any]],
    interactions_rows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS scenes ("
            "scene_id TEXT PRIMARY KEY, "
            "scene_number INTEGER, "
            "heading TEXT, "
            "location TEXT, "
            "time_of_day TEXT, "
            "int_ext TEXT, "
            "characters_json TEXT, "
            "actions_json TEXT, "
            "transitions_json TEXT, "
            "full_text TEXT, "
            "element_ids_json TEXT, "
            "embedding_model TEXT, "
            "embedding_json TEXT, "
            "word_count INTEGER, "
            "dialogue_turns_count INTEGER, "
            "actions_count INTEGER"
            ")"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS dialogue_turns ("
            "scene_id TEXT, "
            "turn_id INTEGER, "
            "speaker TEXT, "
            "text TEXT, "
            "element_ids_json TEXT, "
            "word_count INTEGER, "
            "PRIMARY KEY(scene_id, turn_id), "
            "FOREIGN KEY(scene_id) REFERENCES scenes(scene_id) ON DELETE CASCADE"
            ")"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS characters ("
            "character TEXT PRIMARY KEY, "
            "scenes_json TEXT, "
            "turns INTEGER, "
            "lines INTEGER"
            ")"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS interactions ("
            "character_a TEXT, "
            "character_b TEXT, "
            "scenes_json TEXT, "
            "co_occurrence_scenes INTEGER, "
            "turn_pairs INTEGER, "
            "PRIMARY KEY(character_a, character_b)"
            ")"
        )

        cur.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
            [(k, json.dumps(v, ensure_ascii=False)) for k, v in (meta or {}).items()],
        )

        cur.executemany(
            "INSERT OR REPLACE INTO scenes("
            "scene_id, scene_number, heading, location, time_of_day, int_ext, "
            "characters_json, actions_json, transitions_json, full_text, element_ids_json, "
            "embedding_model, embedding_json, word_count, dialogue_turns_count, actions_count"
            ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.get("scene_id"),
                    r.get("scene_number"),
                    r.get("heading"),
                    r.get("location"),
                    r.get("time_of_day"),
                    r.get("int_ext"),
                    json.dumps(r.get("characters") or [], ensure_ascii=False),
                    json.dumps(r.get("actions") or [], ensure_ascii=False),
                    json.dumps(r.get("transitions") or [], ensure_ascii=False),
                    r.get("full_text"),
                    json.dumps(r.get("element_ids") or [], ensure_ascii=False),
                    r.get("embedding_model"),
                    json.dumps(r.get("embedding") if r.get("embedding") is not None else None, ensure_ascii=False),
                    r.get("word_count"),
                    r.get("dialogue_turns_count"),
                    r.get("actions_count"),
                )
                for r in scenes_rows
            ],
        )

        cur.executemany(
            "INSERT OR REPLACE INTO dialogue_turns(scene_id, turn_id, speaker, text, element_ids_json, word_count) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            [
                (
                    r.get("scene_id"),
                    r.get("turn_id"),
                    r.get("speaker"),
                    r.get("text"),
                    json.dumps(r.get("element_ids") or [], ensure_ascii=False),
                    r.get("word_count"),
                )
                for r in dialogue_rows
            ],
        )

        cur.executemany(
            "INSERT OR REPLACE INTO characters(character, scenes_json, turns, lines) VALUES(?, ?, ?, ?)",
            [
                (
                    r.get("character"),
                    json.dumps(r.get("scenes") or [], ensure_ascii=False),
                    r.get("turns"),
                    r.get("lines"),
                )
                for r in characters_rows
            ],
        )

        cur.executemany(
            "INSERT OR REPLACE INTO interactions(character_a, character_b, scenes_json, co_occurrence_scenes, turn_pairs) "
            "VALUES(?, ?, ?, ?, ?)",
            [
                (
                    r.get("character_a"),
                    r.get("character_b"),
                    json.dumps(r.get("scenes") or [], ensure_ascii=False),
                    r.get("co_occurrence_scenes"),
                    r.get("turn_pairs"),
                )
                for r in interactions_rows
            ],
        )

        try:
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS scenes_fts USING fts5(scene_id, full_text)"
            )
            cur.execute("DELETE FROM scenes_fts")
            cur.executemany(
                "INSERT INTO scenes_fts(scene_id, full_text) VALUES(?, ?)",
                [(r.get("scene_id"), r.get("full_text") or "") for r in scenes_rows],
            )
        except sqlite3.OperationalError:
            pass

        try:
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS dialogue_fts USING fts5(scene_id, turn_id, speaker, text)"
            )
            cur.execute("DELETE FROM dialogue_fts")
            cur.executemany(
                "INSERT INTO dialogue_fts(scene_id, turn_id, speaker, text) VALUES(?, ?, ?, ?)",
                [
                    (r.get("scene_id"), r.get("turn_id"), r.get("speaker"), r.get("text") or "")
                    for r in dialogue_rows
                ],
            )
        except sqlite3.OperationalError:
            pass

        conn.commit()
    finally:
        conn.close()

def build_characters_index(scenes: list[Scene]) -> list[dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for sc in scenes:
        for dt in sc.dialogue:
            c = dt.speaker
            if not c:
                continue
            if c not in idx:
                idx[c] = {"character": c, "scenes": [], "turns": 0, "lines": 0}
            idx[c]["turns"] += 1
            idx[c]["lines"] += len([x for x in dt.text.splitlines() if x.strip()])
            if sc.scene_id not in idx[c]["scenes"]:
                idx[c]["scenes"].append(sc.scene_id)
    return list(idx.values())

def make_next_turn_pairs(scenes: list[Scene], max_context_turns: int = 6) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for sc in scenes:
        turns = sc.dialogue
        if len(turns) < 2:
            continue
        for i in range(1, len(turns)):
            start = max(0, i - max_context_turns)
            ctx = turns[start:i]
            prompt_lines: list[str] = []
            if sc.heading:
                prompt_lines.append(sc.heading)
            if sc.location:
                prompt_lines.append(sc.location)
            prompt_lines.append("الحوار السابق:")
            for t in ctx:
                prompt_lines.append(f"{t.speaker}: {t.text}")
            prompt_lines.append("ما الجملة التالية؟")
            pairs.append(
                {
                    "scene_id": sc.scene_id,
                    "turn_index": i + 1,
                    "prompt": "\n".join(prompt_lines).strip(),
                    "target": f"{turns[i].speaker}: {turns[i].text}",
                }
            )
    return pairs

# ----------------------------
# دالة التحقق من صحة ملف الإدخال
# ----------------------------
def validate_input_file(file_path: str) -> bool:
    """التحقق من صحة ملف الإدخال"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"الملف غير موجود: {file_path}")

    if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
        raise ValueError("حجم الملف كبير جداً (أكثر من 100 ميغابايت)")

    allowed_extensions = ['.txt', '.pdf', '.md', '.json', '.docx']
    if not any(file_path.lower().endswith(ext) for ext in allowed_extensions):
        logger.warning(f"⚠ نوع الملف غير معتاد: {file_path}")

    return True


# ----------------------------
# دالة كتابة ملف آمنة
# ----------------------------
def safe_write_file(path: str, content: str, encoding: str = 'utf-8') -> bool:
    """كتابة ملف مع التحقق من نجاح العملية"""
    try:
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        # التحقق من أن الملف كُتب بنجاح
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return True
        else:
            logger.error(f"✗ فشل في كتابة الملف: {path}")
            return False
    except Exception as e:
        logger.error(f"✗ خطأ في كتابة الملف {path}: {e}")
        return False


# ----------------------------
# دالة تصدير Alpaca المحسّنة مع إثراء السياق
# ----------------------------
def export_enriched_alpaca(scenes: list[Scene], output_path: str, max_context_turns: int = 6) -> list[dict[str, Any]]:
    """
    تصدير بصيغة Alpaca مع السياق المحسّن

    Args:
        scenes: قائمة المشاهد
        output_path: مسار ملف الإخراج
        max_context_turns: عدد الحوارات السابقة للسياق

    Returns:
        قائمة البيانات المصدّرة
    """
    data: list[dict[str, Any]] = []

    for scene in scenes:
        if not scene.dialogue:
            continue

        # استخراج آخر سطر وصفي مهم
        last_action = ContextEnricher.get_last_significant_action(scene.actions)

        # بناء وصف المشهد المحسّن
        scene_setup = ContextEnricher.build_enriched_scene_setup(scene, last_action)

        context_buffer: list[str] = []

        for i, turn in enumerate(scene.dialogue):
            if not turn.text.strip() or not turn.speaker.strip():
                continue

            # تنسيق الإدخال مع السياق الكامل
            full_input = ContextEnricher.format_contextual_input(
                scene, turn, context_buffer, last_action
            )

            data.append({
                "instruction": "أكمل الحوار التالي بناءً على السياق المعطى.",
                "input": full_input,
                "output": turn.text,
                "scene_id": scene.scene_id,
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "time_period": scene.time_period
            })

            # تحديث سجل الحوار
            context_buffer.append(f"{turn.speaker}: {turn.text}")
            if len(context_buffer) > max_context_turns:
                context_buffer = context_buffer[-max_context_turns:]

    # كتابة الملف
    write_jsonl(output_path, data)
    logger.info(f"✓ تم تصدير {len(data)} سجل Alpaca إلى: {output_path}")

    return data


# ----------------------------
# 5) Main - الدالة الرئيسية المحدّثة
# ----------------------------
def main():
    """
    الدالة الرئيسية لنظام الراوي الإصدار 4.0
    تدمج جميع التحسينات: توحيد الكيانات، فلترة الجودة، إثراء السياق، الميتاداتا الزمنية
    """
    import argparse
    import time

    start_time = time.time()

    ap = argparse.ArgumentParser(
        description="نظام الراوي v4.0 - معالجة السيناريوهات العربية"
    )
    ap.add_argument("--input", required=True, help="مسار ملف السيناريو (txt/docx/pdf/...) أو ملف elements.json")
    ap.add_argument("--out_dir", required=True, help="مجلد الإخراج للملفات")
    ap.add_argument("--extractor", default="auto", choices=["auto", "unstructured", "docling"],
                    help="محرك الاستخراج (auto: pdf->docling وإلا unstructured)")
    ap.add_argument("--save_docling_artifacts", action="store_true",
                    help="حفظ مخرجات docling الوسيطة")
    ap.add_argument("--docling_ocr_langs", default="ar,en",
                    help="لغات OCR لـ docling (افتراضي: ar,en)")
    ap.add_argument("--docling_threads", type=int, default=4,
                    help="عدد خيوط docling (افتراضي: 4)")
    ap.add_argument("--use_api_embeddings", action="store_true",
                    help="استخدام API للتضمينات")
    ap.add_argument("--api_work_dir", default="./_unstructured_work",
                    help="مجلد العمل المؤقت لـ API")
    ap.add_argument("--embedder_subtype", default="bedrock")
    ap.add_argument("--embedder_model", default="cohere.embed-multilingual-v3")
    ap.add_argument("--write_sqlite", action="store_true",
                    help="كتابة قاعدة بيانات SQLite")

    # خيارات التحسينات الجديدة (v4.0)
    ap.add_argument("--enable_entity_canonicalization", action="store_true", default=True,
                    help="تفعيل توحيد الكيانات (افتراضي: مفعّل)")
    ap.add_argument("--similarity_threshold", type=float, default=0.85,
                    help="عتبة التشابه لتوحيد الأسماء (افتراضي: 0.85)")
    ap.add_argument("--enable_quality_filter", action="store_true", default=True,
                    help="تفعيل فلترة الجودة (افتراضي: مفعّل)")
    ap.add_argument("--min_words", type=int, default=3,
                    help="الحد الأدنى لعدد الكلمات (افتراضي: 3)")
    ap.add_argument("--sentiment_threshold", type=float, default=0.8,
                    help="عتبة المشاعر للاحتفاظ بالحوارات القصيرة (افتراضي: 0.8)")
    ap.add_argument("--export_alpaca", action="store_true", default=True,
                    help="تصدير بصيغة Alpaca المحسّنة (افتراضي: مفعّل)")

    args = ap.parse_args()

    # إحصائيات العملية الشاملة
    process_stats = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": args.input,
        "output_dir": args.out_dir,
        "stages": {}
    }

    try:
        # ----------------------------
        # المرحلة 1: التحقق من المدخلات
        # ----------------------------
        logger.info("=" * 60)
        logger.info("🚀 بدء معالجة السيناريو - نظام الراوي v4.0")
        logger.info("=" * 60)

        validate_input_file(args.input)
        os.makedirs(args.out_dir, exist_ok=True)
        logger.info(f"✓ ملف الإدخال: {args.input}")
        logger.info(f"✓ مجلد الإخراج: {args.out_dir}")

        # ----------------------------
        # المرحلة 2: استخراج العناصر
        # ----------------------------
        logger.info("\n📄 المرحلة 1: استخراج العناصر من الملف...")
        stage_start = time.time()

# 5) Main - مع تحسينات معالجة الأخطاء والتسجيل (المتطلب 6)
# ----------------------------
def main():
    """
    الدالة الرئيسية لنظام الراوي.

    تنفذ المتطلبات التالية:
    - المتطلب 6.1: تسجيل رسائل خطأ واضحة باللغة العربية
    - المتطلب 6.2: تسجيل إحصائيات العمليات المنجزة
    - المتطلب 6.4: التحقق من وجود البيانات المطلوبة قبل المعالجة
    - المتطلب 6.5: التأكد من نجاح عملية الكتابة قبل المتابعة
    """
    import argparse
    ap = argparse.ArgumentParser(description="نظام الراوي لمعالجة السيناريوهات العربية")
    ap.add_argument("--input", required=True, help="مسار ملف السيناريو (txt/docx/pdf/...) أو قائمة elements.json")
    ap.add_argument("--out_dir", required=True, help="مجلد الإخراج لملفات مجموعة البيانات")
    ap.add_argument("--extractor", default="auto", choices=["auto", "unstructured", "docling"], help="محرك الاستخراج (auto: pdf->docling وإلا unstructured)")
    ap.add_argument("--save_docling_artifacts", action="store_true", help="حفظ مخرجات docling الخام في مجلد الإخراج")
    ap.add_argument("--docling_ocr_langs", default="ar,en", help="لغات OCR لـ docling (افتراضي: ar,en)")
    ap.add_argument("--docling_threads", type=int, default=4, help="عدد خيوط docling (افتراضي: 4)")
    ap.add_argument("--use_api_embeddings", action="store_true", help="استخدام API لإضافة التضمينات")
    ap.add_argument("--api_work_dir", default="./_unstructured_work", help="مجلد العمل المؤقت لوظائف API")
    ap.add_argument("--embedder_subtype", default="bedrock")
    ap.add_argument("--embedder_model", default="cohere.embed-multilingual-v3")
    ap.add_argument("--write_sqlite", action="store_true", help="كتابة قاعدة بيانات SQLite في مجلد الإخراج")
    ap.add_argument("--log_file", default=None, help="مسار ملف السجل (اختياري)")
    ap.add_argument("--verbose", action="store_true", help="تفعيل التسجيل التفصيلي")
    args = ap.parse_args()

    # ----------------------------
    # تهيئة معالجة الأخطاء والتسجيل (المتطلب 6)
    # ----------------------------
    if ERROR_HANDLING_AVAILABLE:
        log_level = LogLevel.DEBUG if args.verbose else LogLevel.INFO
        log_file = args.log_file or os.path.join(args.out_dir, "processing.log") if os.path.exists(args.out_dir) else args.log_file

        logger, error_handler, validator, writer, stats = setup_error_handling(
            log_file=log_file,
            console_output=True,
            log_level=log_level
        )

        logger.info("═" * 60)
        logger.info("نظام الراوي الإصدار 4.0 - معالجة السيناريوهات العربية")
        logger.info("═" * 60)
        logger.info(f"ملف الإدخال: {args.input}")
        logger.info(f"مجلد الإخراج: {args.out_dir}")

        # بدء تتبع الإحصائيات
        stats.start_operation("المعالجة الكاملة")

        # ----------------------------
        # التحقق من صحة المدخلات (المتطلب 6.4)
        # ----------------------------
        logger.info("─" * 40)
        logger.info("التحقق من صحة المدخلات...")

        # التحقق من وجود ملف الإدخال
        input_validation = validator.validate_file_exists(args.input)
        if not input_validation.is_valid:
            for error in input_validation.errors:
                logger.error(error)
            error_handler.record_error(
                error_type="ValidationError",
                message="فشل التحقق من ملف الإدخال",
                details=str(input_validation.errors)
            )
            stats.end_operation("المعالجة الكاملة", items_failed=1)
            return 1

        # التحقق من مجلد الإخراج
        dir_validation = validator.validate_directory(args.out_dir, create_if_missing=True)
        if not dir_validation.is_valid:
            for error in dir_validation.errors:
                logger.error(error)
            return 1

        logger.success("✓ تم التحقق من المدخلات بنجاح")

    else:
        # الوضع الأساسي بدون وحدة معالجة الأخطاء
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"بدء المعالجة: {args.input}")

    # ----------------------------
    # استخراج العناصر
    # ----------------------------
    if ERROR_HANDLING_AVAILABLE:
        logger.info("─" * 40)
        logger.info("استخراج العناصر من الملف...")
        stats.start_operation("استخراج العناصر")

    try:
        docling_langs = [x.strip() for x in (args.docling_ocr_langs or "").split(",") if x.strip()]
        elements, pipeline_meta = elements_from_input(
            input_path=args.input,
            extractor=args.extractor,
            out_dir=args.out_dir,
            save_docling_artifacts=bool(args.save_docling_artifacts),
            docling_ocr_languages=docling_langs or None,
            docling_threads=int(args.docling_threads),
        )

        process_stats["stages"]["extraction"] = {
            "elements_count": len(elements),
            "extractor": pipeline_meta.get("extractor", "unknown"),
            "duration_seconds": round(time.time() - stage_start, 2)
        }
        logger.info(f"✓ تم استخراج {len(elements)} عنصر")

        # ----------------------------
        # المرحلة 3: بناء المشاهد مع الميتاداتا الزمنية
        # ----------------------------
        logger.info("\n🎬 المرحلة 2: بناء هيكل المشاهد...")
        stage_start = time.time()

        scenes = elements_to_scenes(elements)

        # إحصائيات الفترات الزمنية
        time_periods_found = sum(1 for s in scenes if s.time_period != "غير محدد")
        process_stats["stages"]["scene_parsing"] = {
            "scenes_count": len(scenes),
            "time_periods_found": time_periods_found,
            "duration_seconds": round(time.time() - stage_start, 2)
        }
        logger.info(f"✓ تم بناء {len(scenes)} مشهد")
        logger.info(f"✓ تم استخراج الفترة الزمنية لـ {time_periods_found} مشهد")

        # ----------------------------
        # المرحلة 4: توحيد الكيانات (Entity Canonicalization)
        # ----------------------------
        canonicalizer_stats = {}
        if args.enable_entity_canonicalization:
            logger.info("\n👥 المرحلة 3: توحيد أسماء الشخصيات...")
            stage_start = time.time()

            try:
                canonicalizer = EntityCanonicalizer(similarity_threshold=args.similarity_threshold)
                canonicalizer.build_canonical_map(scenes)
                scenes = canonicalizer.apply_normalization(scenes)

                # حفظ سجل الدمج
                merge_log_path = os.path.join(args.out_dir, "entity_merge_log.json")
                canonicalizer.export_merge_log(merge_log_path)

                canonicalizer_stats = canonicalizer.get_stats()
                process_stats["stages"]["entity_canonicalization"] = {
                    **canonicalizer_stats,
                    "duration_seconds": round(time.time() - stage_start, 2)
                }

                if canonicalizer_stats["merges_performed"] > 0:
                    logger.info(f"✓ تم دمج {canonicalizer_stats['merges_performed']} اسم")
                else:
                    logger.info("✓ لا توجد أسماء متشابهة للدمج")

            except Exception as e:
                logger.error(f"✗ فشل توحيد الكيانات: {e}")
                process_stats["stages"]["entity_canonicalization"] = {"error": str(e)}
        else:
            logger.info("\n⏭️ تم تخطي توحيد الكيانات (معطّل)")

        # ----------------------------
        # المرحلة 5: فلترة الجودة
        # ----------------------------
        filter_stats = {}
        original_dialogue_count = sum(len(s.dialogue) for s in scenes)

        if args.enable_quality_filter:
            logger.info("\n🔍 المرحلة 4: فلترة الحوارات منخفضة الجودة...")
            stage_start = time.time()

            try:
                quality_filter = QualityFilter(
                    min_words=args.min_words,
                    high_sentiment_threshold=args.sentiment_threshold
                )
                scenes = quality_filter.filter_scenes(scenes)

                # حفظ سجل الفلترة
                filter_log_path = os.path.join(args.out_dir, "quality_filter_log.json")
                quality_filter.export_filter_log(filter_log_path)

                filter_stats = quality_filter.get_stats()
                process_stats["stages"]["quality_filter"] = {
                    **filter_stats,
                    "duration_seconds": round(time.time() - stage_start, 2)
                }

                if filter_stats["filtered_count"] > 0:
                    logger.info(f"✓ تمت إزالة {filter_stats['filtered_count']} حوار")
                    logger.info(f"✓ تم الاحتفاظ بـ {filter_stats['kept_count']} حوار")
                else:
                    logger.info("✓ جميع الحوارات تجتاز معايير الجودة")

            except Exception as e:
                logger.error(f"✗ فشل فلترة الجودة: {e}")
                process_stats["stages"]["quality_filter"] = {"error": str(e)}
        else:
            logger.info("\n⏭️ تم تخطي فلترة الجودة (معطّلة)")

        # ----------------------------
        # المرحلة 6: التضمينات (اختياري)
        # ----------------------------
        if args.use_api_embeddings:
            logger.info("\n🧠 المرحلة 5: إضافة التضمينات...")
            stage_start = time.time()

            try:
                api_key = os.getenv("UNSTRUCTURED_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("UNSTRUCTURED_API_KEY غير معيّن")

                embed_scenes_via_on_demand_jobs(
                    scenes=scenes,
                    api_key=api_key,
                    work_dir=args.api_work_dir,
                    batch_size=10,
                    embedder_subtype=args.embedder_subtype,
                    embedder_model=args.embedder_model,
                )

                embedded_count = sum(1 for s in scenes if s.embedding is not None)
                process_stats["stages"]["embeddings"] = {
                    "embedded_scenes": embedded_count,
                    "duration_seconds": round(time.time() - stage_start, 2)
                }
                logger.info(f"✓ تم تضمين {embedded_count} مشهد")

            except Exception as e:
                logger.error(f"✗ فشل إضافة التضمينات: {e}")
                process_stats["stages"]["embeddings"] = {"error": str(e)}

        # ----------------------------
        # المرحلة 7: بناء مجموعات البيانات
        # ----------------------------
        logger.info("\n📊 المرحلة 6: بناء مجموعات البيانات...")
        stage_start = time.time()

        # بناء صفوف البيانات مع الميتاداتا الزمنية
        scenes_rows: list[dict[str, Any]] = []
        dialogue_rows: list[dict[str, Any]] = []

        for sc in scenes:
            scenes_rows.append({

        if ERROR_HANDLING_AVAILABLE:
            # التحقق من العناصر المستخرجة
            elements_validation = validator.validate_elements(elements)
            if not elements_validation.is_valid:
                for error in elements_validation.errors:
                    logger.error(error)
                return 1

            stats.end_operation("استخراج العناصر", items_processed=len(elements))
            logger.success(f"✓ تم استخراج {len(elements)} عنصر")

    except Exception as e:
        if ERROR_HANDLING_AVAILABLE:
            error_handler.handle_exception(e, "استخراج العناصر")
            stats.end_operation("استخراج العناصر", items_failed=1)
            logger.critical(f"فشل في استخراج العناصر: {str(e)}")
        else:
            print(f"خطأ: {str(e)}")
        raise

    # ----------------------------
    # تحليل المشاهد
    # ----------------------------
    if ERROR_HANDLING_AVAILABLE:
        logger.info("─" * 40)
        logger.info("تحليل المشاهد...")
        stats.start_operation("تحليل المشاهد")

    try:
        scenes = elements_to_scenes(elements)

        if ERROR_HANDLING_AVAILABLE:
            scenes_validation = validator.validate_scenes(scenes)
            if not scenes_validation.is_valid:
                for error in scenes_validation.errors:
                    logger.error(error)
            for warning in scenes_validation.warnings:
                logger.info(warning)

            stats.end_operation("تحليل المشاهد", items_processed=len(scenes))
            logger.success(f"✓ تم تحليل {len(scenes)} مشهد")

    except Exception as e:
        if ERROR_HANDLING_AVAILABLE:
            error_handler.handle_exception(e, "تحليل المشاهد")
            stats.end_operation("تحليل المشاهد", items_failed=1)
        raise

    # ----------------------------
    # استخراج الميتاداتا الزمنية (المتطلب 4)
    # ----------------------------
    if ERROR_HANDLING_AVAILABLE:
        logger.info("─" * 40)
        logger.info("استخراج الميتاداتا الزمنية...")
        stats.start_operation("استخراج الميتاداتا الزمنية")

    # تطبيق استخراج الميتاداتا الزمنية
    temporal_extractor = TemporalMetadataExtractor()
    scenes = temporal_extractor.apply_to_scenes(scenes)
    temporal_stats = temporal_extractor.get_extraction_stats()

    if ERROR_HANDLING_AVAILABLE:
        stats.end_operation(
            "استخراج الميتاداتا الزمنية",
            items_processed=temporal_stats["found_years"],
            details=temporal_stats
        )
        logger.success(f"✓ تم استخراج الفترات الزمنية: {temporal_stats['found_years']} وُجدت، {temporal_stats['inherited']} موروثة")

    # ----------------------------
    # التضمينات الاختيارية
    # ----------------------------
    # Optional: API embeddings
    if args.use_api_embeddings:
        if ERROR_HANDLING_AVAILABLE:
            logger.info("─" * 40)
            logger.info("إضافة التضمينات عبر API...")
            stats.start_operation("إضافة التضمينات")

        api_key = os.getenv("UNSTRUCTURED_API_KEY", "").strip()
        if not api_key:
            error_msg = "متغير البيئة UNSTRUCTURED_API_KEY غير معين"
            if ERROR_HANDLING_AVAILABLE:
                logger.error(error_msg)
                error_handler.record_error("ConfigError", error_msg)
            raise RuntimeError(error_msg)

        try:
            embed_scenes_via_on_demand_jobs(
                scenes=scenes,
                api_key=api_key,
                work_dir=args.api_work_dir,
                batch_size=10,
                embedder_subtype=args.embedder_subtype,
                embedder_model=args.embedder_model,
            )
            if ERROR_HANDLING_AVAILABLE:
                stats.end_operation("إضافة التضمينات", items_processed=len(scenes))
                logger.success("✓ تم إضافة التضمينات")

        except Exception as e:
            if ERROR_HANDLING_AVAILABLE:
                # المتابعة عند فشل API (المتطلب 6.3)
                error_handler.handle_exception(e, "إضافة التضمينات", continue_on_error=True)
                stats.end_operation("إضافة التضمينات", items_failed=len(scenes))
                logger.warning("تم المتابعة بدون التضمينات")

    # ----------------------------
    # تجهيز البيانات للتصدير
    # ----------------------------
    if ERROR_HANDLING_AVAILABLE:
        logger.info("─" * 40)
        logger.info("تجهيز البيانات للتصدير...")
        stats.start_operation("تجهيز البيانات")

    scenes_rows: list[dict[str, Any]] = []
    dialogue_rows: list[dict[str, Any]] = []
    for sc in scenes:
        scenes_rows.append(
            {
                "scene_id": sc.scene_id,
                "scene_number": sc.scene_number,
                "heading": sc.heading,
                "location": sc.location,
                "time_of_day": sc.time_of_day,
                "int_ext": sc.int_ext,
                "time_period": sc.time_period,  # حقل جديد
                "time_period": sc.time_period,  # حقل الفترة الزمنية (المتطلب 4)
                "time_period": sc.time_period,  # حقل جديد للفترة الزمنية
                "characters": sc.characters,
                "actions": sc.actions,
                "transitions": sc.transitions,
                "full_text": sc.full_text,
                "element_ids": sc.element_ids,
                "embedding_model": sc.embedding_model,
                "embedding": sc.embedding,
                "word_count": len((sc.full_text or "").split()),
                "dialogue_turns_count": len(sc.dialogue),
                "actions_count": len(sc.actions),
            })

            for dt in sc.dialogue:
                dialogue_rows.append({
                    "scene_id": dt.scene_id,
                    "turn_id": dt.turn_id,
                    "speaker": dt.speaker,
                    "text": dt.text,
                    "element_ids": dt.element_ids,
                    "word_count": len((dt.text or "").split()),
                })

        characters_rows = build_characters_index(scenes)
        pairs_rows = make_next_turn_pairs(scenes)
        interactions_rows = build_interactions_index(scenes)
        speaker_id_rows = make_speaker_id_pairs(scenes)

        process_stats["stages"]["dataset_building"] = {
            "duration_seconds": round(time.time() - stage_start, 2)
        }

        # ----------------------------
        # المرحلة 8: كتابة الملفات
        # ----------------------------
        logger.info("\n💾 المرحلة 7: كتابة الملفات...")
        stage_start = time.time()

        files_written = []

        # كتابة ملفات JSONL الأساسية
        write_jsonl(os.path.join(args.out_dir, "scenes.jsonl"), scenes_rows)
        files_written.append("scenes.jsonl")

        write_jsonl(os.path.join(args.out_dir, "dialogue_turns.jsonl"), dialogue_rows)
        files_written.append("dialogue_turns.jsonl")

        write_jsonl(os.path.join(args.out_dir, "characters.jsonl"), characters_rows)
        files_written.append("characters.jsonl")

        write_jsonl(os.path.join(args.out_dir, "next_turn_pairs.jsonl"), pairs_rows)
        files_written.append("next_turn_pairs.jsonl")

        write_jsonl(os.path.join(args.out_dir, "character_interactions.jsonl"), interactions_rows)
        files_written.append("character_interactions.jsonl")

        write_jsonl(os.path.join(args.out_dir, "speaker_id_pairs.jsonl"), speaker_id_rows)
        files_written.append("speaker_id_pairs.jsonl")

        # تصدير Alpaca المحسّن مع إثراء السياق
        if args.export_alpaca:
            alpaca_path = os.path.join(args.out_dir, "alpaca_enriched.jsonl")
            alpaca_data = export_enriched_alpaca(scenes, alpaca_path)
            files_written.append("alpaca_enriched.jsonl")
            process_stats["alpaca_records"] = len(alpaca_data)

        # حفظ العناصر الخام
        with open(os.path.join(args.out_dir, "elements.local.json"), "w", encoding="utf-8") as f:
            json.dump(elements, f, ensure_ascii=False, indent=2)
        files_written.append("elements.local.json")

        # قاعدة بيانات SQLite
        if args.write_sqlite:
            db_path = os.path.join(args.out_dir, "screenplay_dataset.sqlite")
            write_sqlite_db(
                db_path=db_path,
                scenes_rows=scenes_rows,
                dialogue_rows=dialogue_rows,
                characters_rows=characters_rows,
                interactions_rows=interactions_rows,
                meta=pipeline_meta,
            )
            files_written.append("screenplay_dataset.sqlite")

        process_stats["stages"]["file_writing"] = {
            "files_written": files_written,
            "duration_seconds": round(time.time() - stage_start, 2)
        }

        # ----------------------------
        # المرحلة 9: حفظ الإحصائيات والتقرير النهائي
        # ----------------------------
        total_time = time.time() - start_time
        process_stats["total_duration_seconds"] = round(total_time, 2)
        process_stats["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # الإحصائيات النهائية
        process_stats["final_stats"] = {
            "scenes": len(scenes_rows),
            "dialogue_turns": len(dialogue_rows),
            "characters": len(characters_rows),
            "next_turn_pairs": len(pairs_rows),
            "interactions": len(interactions_rows),
            "speaker_id_pairs": len(speaker_id_rows),
            "time_periods_extracted": time_periods_found,
            "entities_merged": canonicalizer_stats.get("merges_performed", 0),
            "dialogues_filtered": filter_stats.get("filtered_count", 0)
        }

        # حفظ ملف الإحصائيات
        stats_path = os.path.join(args.out_dir, "processing_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(process_stats, f, ensure_ascii=False, indent=2)

        # ----------------------------
        # الطباعة النهائية
        # ----------------------------
        logger.info("\n" + "=" * 60)
        logger.info("✅ اكتملت المعالجة بنجاح!")
        logger.info("=" * 60)
        print("\n📈 الإحصائيات النهائية:")
        print(f"   • المشاهد: {len(scenes_rows)}")
        print(f"   • الحوارات: {len(dialogue_rows)}")
        print(f"   • الشخصيات: {len(characters_rows)}")
        print(f"   • أزواج الحوار التالي: {len(pairs_rows)}")
        print(f"   • التفاعلات: {len(interactions_rows)}")
        print(f"   • أزواج تحديد المتحدث: {len(speaker_id_rows)}")

        print("\n🆕 تحسينات v4.0:")
        print(f"   • الفترات الزمنية المستخرجة: {time_periods_found}")
        print(f"   • الأسماء المدمجة: {canonicalizer_stats.get('merges_performed', 0)}")
        print(f"   • الحوارات المفلترة: {filter_stats.get('filtered_count', 0)}")

        print(f"\n⏱️ الوقت الإجمالي: {total_time:.2f} ثانية")
        print(f"📁 مجلد الإخراج: {args.out_dir}")
        print(f"📄 عدد الملفات: {len(files_written)}")

    except FileNotFoundError as e:
        logger.error(f"✗ خطأ: {e}")
        raise
    except ValueError as e:
        logger.error(f"✗ خطأ في القيمة: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ خطأ غير متوقع: {e}")
        # حفظ حالة الخطأ
        error_stats = {
            "error": str(e),
            "error_type": type(e).__name__,
            "stages_completed": list(process_stats.get("stages", {}).keys())
        }
        error_path = os.path.join(args.out_dir, "error_log.json")
        try:
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(error_stats, f, ensure_ascii=False, indent=2)
        except:
            pass
        raise
    characters_rows = build_characters_index(scenes)
    pairs_rows = make_next_turn_pairs(scenes)
    interactions_rows = build_interactions_index(scenes)
    speaker_id_rows = make_speaker_id_pairs(scenes)

    if ERROR_HANDLING_AVAILABLE:
        stats.end_operation("تجهيز البيانات", items_processed=len(scenes_rows) + len(dialogue_rows))
        logger.success("✓ تم تجهيز البيانات")

    # ----------------------------
    # كتابة الملفات (المتطلب 6.5: ضمان نجاح الكتابة)
    # ----------------------------
    if ERROR_HANDLING_AVAILABLE:
        logger.info("─" * 40)
        logger.info("كتابة ملفات الإخراج...")
        stats.start_operation("كتابة الملفات")

        files_to_write = [
            ("scenes.jsonl", scenes_rows),
            ("dialogue_turns.jsonl", dialogue_rows),
            ("characters.jsonl", characters_rows),
            ("next_turn_pairs.jsonl", pairs_rows),
            ("character_interactions.jsonl", interactions_rows),
            ("speaker_id_pairs.jsonl", speaker_id_rows),
        ]

        write_success_count = 0
        write_fail_count = 0

        for filename, rows in files_to_write:
            path = os.path.join(args.out_dir, filename)
            result = writer.write_jsonl(path, rows)
            if result.success:
                write_success_count += 1
                logger.info(f"  ✓ {filename}: {len(rows)} سجل ({result.bytes_written} بايت)")
            else:
                write_fail_count += 1
                logger.error(f"  ✗ فشل في كتابة {filename}: {result.error}")

        # كتابة العناصر المحلية
        elements_result = writer.write_json(
            os.path.join(args.out_dir, "elements.local.json"),
            elements,
            indent=2
        )
        if elements_result.success:
            write_success_count += 1
            logger.info(f"  ✓ elements.local.json: {len(elements)} عنصر")
        else:
            write_fail_count += 1
            logger.error(f"  ✗ فشل في كتابة elements.local.json")

        stats.end_operation(
            "كتابة الملفات",
            items_processed=write_success_count,
            items_failed=write_fail_count
        )

        # التحقق من نجاح جميع عمليات الكتابة (المتطلب 6.5)
        if not writer.all_successful():
            logger.warning("تحذير: بعض عمليات الكتابة فشلت")
            failed_writes = writer.get_failed_writes()
            for fw in failed_writes:
                logger.error(f"  - {fw.path}: {fw.error}")

    else:
        # الوضع الأساسي
        write_jsonl(os.path.join(args.out_dir, "scenes.jsonl"), scenes_rows)
        write_jsonl(os.path.join(args.out_dir, "dialogue_turns.jsonl"), dialogue_rows)
        write_jsonl(os.path.join(args.out_dir, "characters.jsonl"), characters_rows)
        write_jsonl(os.path.join(args.out_dir, "next_turn_pairs.jsonl"), pairs_rows)
        write_jsonl(os.path.join(args.out_dir, "character_interactions.jsonl"), interactions_rows)
        write_jsonl(os.path.join(args.out_dir, "speaker_id_pairs.jsonl"), speaker_id_rows)

        with open(os.path.join(args.out_dir, "elements.local.json"), "w", encoding="utf-8") as f:
            json.dump(elements, f, ensure_ascii=False, indent=2)

    # ----------------------------
    # كتابة قاعدة بيانات SQLite
    # ----------------------------
    if args.write_sqlite:
        if ERROR_HANDLING_AVAILABLE:
            logger.info("─" * 40)
            logger.info("كتابة قاعدة بيانات SQLite...")
            stats.start_operation("كتابة SQLite")

        try:
            db_path = os.path.join(args.out_dir, "screenplay_dataset.sqlite")
            write_sqlite_db(
                db_path=db_path,
                scenes_rows=scenes_rows,
                dialogue_rows=dialogue_rows,
                characters_rows=characters_rows,
                interactions_rows=interactions_rows,
                meta=pipeline_meta,
            )

            if ERROR_HANDLING_AVAILABLE:
                stats.end_operation("كتابة SQLite", items_processed=1)
                logger.success(f"✓ تم كتابة قاعدة البيانات: {db_path}")

        except Exception as e:
            if ERROR_HANDLING_AVAILABLE:
                error_handler.handle_exception(e, "كتابة SQLite")
                stats.end_operation("كتابة SQLite", items_failed=1)
            raise

    # ----------------------------
    # إنهاء وطباعة الملخص (المتطلب 6.2: تسجيل الإحصائيات)
    # ----------------------------
    if ERROR_HANDLING_AVAILABLE:
        stats.end_operation(
            "المعالجة الكاملة",
            items_processed=len(scenes),
            details={
                "scenes": len(scenes_rows),
                "dialogue_turns": len(dialogue_rows),
                "characters": len(characters_rows),
                "next_turn_pairs": len(pairs_rows),
                "interactions": len(interactions_rows),
                "speaker_id_pairs": len(speaker_id_rows),
            }
        )

        # إضافة إحصائيات عامة
        stats.add_global_stat("إجمالي_المشاهد", len(scenes_rows))
        stats.add_global_stat("إجمالي_الحوارات", len(dialogue_rows))
        stats.add_global_stat("إجمالي_الشخصيات", len(characters_rows))
        stats.add_global_stat("إجمالي_الكلمات", sum(r.get("word_count", 0) for r in dialogue_rows))

        logger.info("═" * 60)
        logger.info("ملخص المعالجة")
        logger.info("═" * 60)
        stats.print_summary()

        # تصدير الإحصائيات
        stats_path = os.path.join(args.out_dir, "processing_stats.json")
        if stats.export_stats(stats_path):
            logger.info(f"تم تصدير الإحصائيات: {stats_path}")

        # تصدير سجلات الأخطاء إذا وجدت
        if error_handler.has_critical_errors():
            errors_path = os.path.join(args.out_dir, "errors.json")
            with open(errors_path, 'w', encoding='utf-8') as f:
                json.dump(error_handler.get_error_summary(), f, ensure_ascii=False, indent=2)
            logger.warning(f"تم تصدير سجل الأخطاء: {errors_path}")

        logger.info("═" * 60)
        logger.info("✓ اكتملت المعالجة بنجاح")
        logger.info("═" * 60)

    else:
        print("\n" + "=" * 50)
        print("تم الانتهاء")
        print("=" * 50)

    print(f"- المشاهد: {len(scenes_rows)}")
    print(f"- الحوارات: {len(dialogue_rows)}")
    print(f"- الشخصيات: {len(characters_rows)}")
    print(f"- أزواج الدور التالي: {len(pairs_rows)}")
    print(f"- التفاعلات: {len(interactions_rows)}")
    print(f"- أزواج تحديد المتحدث: {len(speaker_id_rows)}")
    print(f"- الميتاداتا الزمنية: وُجدت={temporal_stats['found_years']}, موروثة={temporal_stats['inherited']}")
    print(f"مجلد الإخراج: {args.out_dir}")

    return 0
    print("DONE")
    print(f"- scenes: {len(scenes_rows)}")
    print(f"- dialogue turns: {len(dialogue_rows)}")
    print(f"- characters: {len(characters_rows)}")
    print(f"- next-turn pairs: {len(pairs_rows)}")
    print(f"- interactions: {len(interactions_rows)}")
    print(f"- speaker-id pairs: {len(speaker_id_rows)}")
    print(f"- temporal metadata: found={temporal_stats['found_years']}, inherited={temporal_stats['inherited']}")
    print(f"Output dir: {args.out_dir}")

if __name__ == "__main__":
    main()
