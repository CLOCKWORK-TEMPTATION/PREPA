import os
import re
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, List
from pathlib import Path
from collections import defaultdict

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('screenplay_dataset.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- Fail-Safe Imports ---
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline as hf_pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    SentenceTransformer = None
    hf_pipeline = None

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    genai_types = None
    print("WARNING: مكتبة google-genai غير مثبتة. pip install google-genai")

# استيراد وحدة توحيد الكيانات
try:
    from entity_canonicalizer import EntityCanonicalizer, canonicalize_scenes, SIMILARITY_AVAILABLE
    CANONICALIZER_AVAILABLE = True
except ImportError:
    CANONICALIZER_AVAILABLE = False
    SIMILARITY_AVAILABLE = False
    print("WARNING: وحدة entity_canonicalizer غير متوفرة")

# استيراد وحدات الراوي 4.0 (إثراء السياق وفلترة الجودة)
try:
    from al_rawi_v4 import ContextEnricher, QualityFilter
    from al_rawi_v4 import count_arabic_words as al_rawi_count_words
    from al_rawi_v4 import is_significant_action
    AL_RAWI_V4_AVAILABLE = True
except ImportError:
    AL_RAWI_V4_AVAILABLE = False
    ContextEnricher = None
    QualityFilter = None
    print("WARNING: وحدات al_rawi_v4 غير متوفرة")

# Docling لمعالجة ملفات PDF
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        EasyOcrOptions,
    )
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("WARNING: مكتبة Docling غير مثبتة. pip install docling")

# قراءة API Keys من متغيرات البيئة
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
class Config:
    CONTEXT_WINDOW_SIZE = 5
    MIN_DIALOGUE_LENGTH = 2
    EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'
    SENTIMENT_MODEL = 'CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment'
    GEMINI_MODEL = 'gemini-3-flash-preview'  # New SDK format
    
    SCENE_PATTERN = re.compile(r"^\s*(?:مشهد|م\.|Scene)\s*[:\-]?\s*(\d+)|^(?:داخلي|خارجي|INT\.|EXT\.)", re.IGNORECASE)
    SPEAKER_PATTERN = re.compile(r"^\s*([أ-يa-zA-Z\s]{2,25})\s*(?::)?\s*$")
    TRANSITIONS = {"قطع", "كات", "CUT", "FADE OUT", "FADE IN", "إظلام", "تلاشي"}

# ---------------------------------------------------------
# Data Models
# ---------------------------------------------------------
@dataclass
class DialogueTurn:
    scene_id: str
    turn_id: int
    speaker: str
    text: str
    normalized_text: str = ""
    sentiment: str = "unknown"
    sentiment_score: float = 0.0

@dataclass
class Scene:
    scene_id: str
    scene_number: int
    heading: str
    location: str
    time_of_day: str
    int_ext: str
    time_period: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    dialogue: List[DialogueTurn] = field(default_factory=list)
    characters: List[str] = field(default_factory=list)
    full_text: str = ""
    embedding: Optional[List[float]] = None

# ---------------------------------------------------------
# Arabic Text Utilities
# ---------------------------------------------------------
def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # Remove diacritics
    text = re.sub(r'\u0640+', '', text)  # Remove tatweel
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[يى]', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text.strip()

def count_arabic_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r'[\u0600-\u06FF]+', text))

# ---------------------------------------------------------
# File Ingestion
# ---------------------------------------------------------
class TextFileIngestor:
    """قراءة ملفات TXT مباشرة"""
    def process(self, file_path: str) -> List[str]:
        logger.info(f"جاري قراءة الملف النصي: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.readlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                return f.readlines()
        except Exception as e:
            logger.error(f"فشل قراءة الملف: {e}")
            return []

class DoclingIngestor:
    """قراءة ملفات PDF باستخدام Docling مع OCR للعربية"""
    def __init__(self):
        if not DOCLING_AVAILABLE:
            raise RuntimeError("مكتبة Docling غير مثبتة!")
        
        # إعدادات OCR متقدمة للعربية
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions(lang=["ar", "en"])
        pipeline_options.do_table_structure = False
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.AUTO
        )
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def process(self, file_path: str) -> List[str]:
        logger.info(f"جاري معالجة ملف PDF بصرياً: {file_path}")
        try:
            result = self.converter.convert(file_path)
            md_text = result.document.export_to_markdown()
            return md_text.split('\n')
        except Exception as e:
            logger.error(f"فشل استخراج النص من PDF: {e}")
            return []

def get_ingestor(file_path: str):
    """اختيار المعالج المناسب حسب نوع الملف"""
    ext = Path(file_path).suffix.lower()
    if ext in ['.txt', '.md']:
        return TextFileIngestor()
    elif ext == '.pdf':
        if not DOCLING_AVAILABLE:
            raise RuntimeError("مكتبة Docling غير مثبتة لمعالجة PDF! pip install docling")
        return DoclingIngestor()
    else:
        return TextFileIngestor()

# ---------------------------------------------------------
# Screenplay Parser
# ---------------------------------------------------------
class ScreenplayParser:
    def __init__(self):
        self.normalizer = self
    
    def normalize(self, text: str) -> str:
        return normalize_arabic(text)
    
    def _clean_markdown(self, line: str) -> str:
        """تنظيف السطر من علامات Markdown"""
        return line.replace('**', '').replace('###', '').replace('##', '').replace('#', '').strip()
    
    def parse(self, lines: List[str]) -> List[Scene]:
        scenes: List[Scene] = []
        current_scene: Optional[Scene] = None
        current_speaker: Optional[str] = None
        current_turn_lines: List[str] = []
        turn_counter = 0

        def flush_turn():
            nonlocal current_speaker, current_turn_lines, turn_counter
            if current_scene and current_speaker and current_turn_lines:
                full_text = " ".join(current_turn_lines).strip()
                if full_text:
                    turn_counter += 1
                    # تخزين النص الأصلي والمطبع
                    norm_text = self.normalizer.normalize(full_text)
                    current_scene.dialogue.append(DialogueTurn(
                        scene_id=current_scene.scene_id,
                        turn_id=turn_counter,
                        speaker=current_speaker,
                        text=full_text,
                        normalized_text=norm_text
                    ))
                    if current_speaker not in current_scene.characters:
                        current_scene.characters.append(current_speaker)
            current_speaker = None
            current_turn_lines = []

        def finalize_scene(scene: Scene):
            parts = [scene.heading] + scene.actions
            for dt in scene.dialogue:
                parts.append(f"{dt.speaker}: {dt.text}")
            scene.full_text = "\n".join(parts)

        # الحلقة الرئيسية
        for raw_line in lines:
            line = self._clean_markdown(raw_line)
            if not line: continue

            # 1. اكتشاف بداية المشهد
            scene_match = Config.SCENE_PATTERN.search(line)
            # شرط إضافي: السطر ليس طويلاً جداً ليكون وصفاً
            is_header = scene_match and len(line) < 60

            if is_header:
                if current_scene:
                    flush_turn()
                    finalize_scene(current_scene)
                    scenes.append(current_scene)
                
                # استخراج الرقم إن وجد
                num_match = re.search(r'\d+', line)
                num = int(num_match.group(0)) if num_match else len(scenes) + 1
                
                # استخراج الميتاداتا
                time_val = next((t for t in ["ليل", "نهار", "مساء", "صباح"] if t in line), "غير محدد")
                loc_val = re.sub(r'(مشهد|م\.|Scene|\d+|ليل|نهار|خارجي|داخلي|[\-\.])', '', line).strip()
                
                # استخراج السنة من العنوان
                year_match = re.search(r'\b(19|20|21)\d{2}\b', line)
                time_period_val = year_match.group(0) if year_match else None
                
                current_scene = Scene(
                    scene_id=f"S{num:04d}",
                    scene_number=num,
                    heading=line,
                    location=loc_val or "موقع غير محدد",
                    time_of_day=time_val,
                    int_ext="داخلي" if "داخلي" in line else "خارجي",
                    time_period=time_period_val
                )
                turn_counter = 0
                continue

            if current_scene is None: continue

            # 2. اكتشاف الانتقالات (Transitions)
            if line in Config.TRANSITIONS:
                flush_turn()
                current_scene.actions.append(f"[TRANSITION: {line}]")
                continue

            # 3. اكتشاف المتحدث (Heuristics + Regex)
            speaker_match = Config.SPEAKER_PATTERN.match(line)
            # الشروط: يطابق النمط + ليس طويلاً + ليس فعلاً بين قوسين
            if speaker_match and len(line.split()) <= 4 and not line.startswith('('):
                potential_name = speaker_match.group(1).strip()
                # تجاهل الكلمات الشائعة التي قد تشبه الأسماء
                if potential_name not in ["صوت", "تكملة", "تابع"]:
                    flush_turn()
                    current_speaker = potential_name
                    continue

            # 4. محتوى الحوار أو الوصف
            if current_speaker:
                # إذا كان السطر بين قوسين، نعتبره "Parenthetical" (وصف طريقة نطق) ولا نضيفه للحوار الصافي
                if line.startswith('(') and line.endswith(')'):
                    continue 
                current_turn_lines.append(line)
            else:
                current_scene.actions.append(line)

        # إضافة آخر مشهد
        if current_scene:
            flush_turn()
            finalize_scene(current_scene)
            scenes.append(current_scene)

        # آلية وراثة الفترة الزمنية بين المشاهد المتتالية
        self._inherit_time_periods(scenes)

        return scenes

    def _inherit_time_periods(self, scenes: List[Scene]):
        """وراثة الفترة الزمنية من المشاهد السابقة للمشاهد التي لا تحتوي على فترة محددة"""
        current_time_period = None
        for scene in scenes:
            if scene.time_period:
                current_time_period = scene.time_period
            elif current_time_period:
                scene.time_period = current_time_period

# ---------------------------------------------------------
# 6. طبقة الإثراء الذكي (Enrichment Layer)
# ---------------------------------------------------------
class AIEnricher:
    """
    طبقة الإثراء الذكي للمشاهد والحوارات

    تشمل:
    - توحيد أسماء الشخصيات (Entity Canonicalization)
    - توليد التضمينات (Embeddings)
    - تحليل المشاعر (Sentiment Analysis)
    """

    def __init__(self, use_gpu=True, similarity_threshold: float = 0.85):
        """
        تهيئة طبقة الإثراء

        Args:
            use_gpu: استخدام GPU إذا كان متوفراً
            similarity_threshold: عتبة التشابه لتوحيد الأسماء (افتراضي: 85%)
        """
        self.embedder = None
        self.sentiment_analyzer = None
        self.canonicalizer = None
        self.canonicalization_stats = {}

        # تهيئة موحد الكيانات
        if CANONICALIZER_AVAILABLE and SIMILARITY_AVAILABLE:
            try:
                self.canonicalizer = EntityCanonicalizer(similarity_threshold=similarity_threshold)
                logger.info(f"تم تهيئة موحد الكيانات (عتبة التشابه: {similarity_threshold:.0%})")
            except Exception as e:
                logger.warning(f"فشل تهيئة موحد الكيانات: {e}")
        else:
            logger.warning("وحدة توحيد الكيانات غير متوفرة - تخطي توحيد الأسماء")

        if ML_AVAILABLE:
            try:
                logger.info("تحميل نموذج Embeddings (E5-Small)...")
                self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL, device='cuda' if use_gpu else 'cpu')

                logger.info("تحميل نموذج تحليل المشاعر (CamelBERT)...")
                self.sentiment_analyzer = hf_pipeline("text-classification", model=Config.SENTIMENT_MODEL, device=0 if use_gpu else -1)
            except Exception as e:
                logger.warning(f"فشل تحميل النماذج: {e}")

    def canonicalize_entities(self, scenes: List[Scene], merge_log_path: Optional[Path] = None) -> List[Scene]:
        """
        توحيد أسماء الشخصيات في المشاهد

        يقوم ببناء قاموس التطبيع وتطبيقه على جميع الحوارات

        Args:
            scenes: قائمة المشاهد
            merge_log_path: مسار حفظ سجل الدمج (اختياري)

        Returns:
            المشاهد بعد توحيد الأسماء
        """
        if not self.canonicalizer:
            logger.info("موحد الكيانات غير متوفر - تخطي توحيد الأسماء")
            return scenes

        logger.info("بدء توحيد أسماء الشخصيات...")

        # بناء قاموس التطبيع
        canonical_map = self.canonicalizer.build_canonical_map(scenes)

        if canonical_map:
            logger.info(f"تم العثور على {len(canonical_map)} اسم للتوحيد")

            # تطبيق التوحيد
            scenes = self.canonicalizer.apply_normalization(scenes)

            # حفظ سجل الدمج
            if merge_log_path:
                self.canonicalizer.export_merge_log(merge_log_path)

            # حفظ الإحصائيات
            self.canonicalization_stats = self.canonicalizer.get_statistics()
            logger.info(f"إحصائيات التوحيد: {self.canonicalization_stats}")
        else:
            logger.info("لا توجد أسماء متشابهة للتوحيد")

        return scenes

    def enrich(self, scenes: List[Scene], canonicalize: bool = True, merge_log_path: Optional[Path] = None) -> List[Scene]:
        """
        إثراء المشاهد بجميع التحسينات

        يشمل: توحيد الأسماء، التضمينات، تحليل المشاعر

        Args:
            scenes: قائمة المشاهد
            canonicalize: تطبيق توحيد الأسماء (افتراضي: True)
            merge_log_path: مسار حفظ سجل دمج الأسماء

        Returns:
            المشاهد بعد الإثراء
        """
        # 1. توحيد أسماء الشخصيات (أولاً قبل أي معالجة أخرى)
        if canonicalize:
            scenes = self.canonicalize_entities(scenes, merge_log_path)

        # 2. توليد التضمينات
        if self.embedder:
            logger.info("بدء توليد التضمينات (Embeddings)...")
            texts = [f"passage: {s.full_text[:2000]}" for s in scenes]
            embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=16)
            for i, scene in enumerate(scenes):
                scene.embedding = embeddings[i].tolist()

        # 3. تحليل المشاعر
        if self.sentiment_analyzer:
            logger.info("بدء تحليل مشاعر الحوارات...")
            for scene in scenes:
                for turn in scene.dialogue:
                    try:
                        # تحليل النصوص القصيرة فقط لتوفير الوقت
                        res = self.sentiment_analyzer(turn.text[:400])[0]
                        turn.sentiment = res['label']
                        turn.sentiment_score = res['score']
                    except:
                        pass

        return scenes

    def get_canonicalization_stats(self) -> dict:
        """
        الحصول على إحصائيات توحيد الأسماء

        Returns:
            قاموس الإحصائيات
        """
        return self.canonicalization_stats

    def build_social_graph(self, scenes: List[Scene]):
        """Build character interaction graph"""
        if not NETWORKX_AVAILABLE:
            logger.warning("مكتبة networkx غير متوفرة - تخطي بناء شبكة العلاقات")
            return None
        G = nx.Graph()
        import itertools
        for scene in scenes:
            chars = list(set(scene.characters))
            if len(chars) < 2: continue
            for c1, c2 in itertools.combinations(chars, 2):
                if G.has_edge(c1, c2):
                    G[c1][c2]['weight'] += 1
                else:
                    G.add_edge(c1, c2, weight=1)
        return G

# ---------------------------------------------------------
# 7. محلل Gemini المتقدم (Advanced AI Analysis)
# ---------------------------------------------------------
class GeminiAnalyzer:
    def __init__(self):
        self.client = None
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                # New google-genai SDK (December 2025+)
                self.client = genai.Client(api_key=GEMINI_API_KEY)
                logger.info(f"✅ تم تهيئة Gemini ({Config.GEMINI_MODEL}) بنجاح")
            except Exception as e:
                logger.error(f"فشل تهيئة Gemini: {e}")
        else:
            logger.warning("Gemini غير متوفر - تأكد من وجود API Key")
    
    def _call_gemini(self, prompt: str) -> str:
        """استدعاء آمن لـ Gemini مع معالجة الأخطاء"""
        if not self.client:
            return "Gemini غير متوفر"
        try:
            response = self.client.models.generate_content(
                model=Config.GEMINI_MODEL,
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"خطأ في Gemini: {e}")
            return f"خطأ: {e}"
    
    def analyze_sentiment_deep(self, scenes: List[Scene]) -> dict:
        """تحليل عميق للمشاعر عبر السيناريو"""
        if not self.client:
            return {}
        
        logger.info("🎭 بدء التحليل العميق للمشاعر بواسطة Gemini...")
        
        # تجميع عينة من الحوارات
        sample_dialogues = []
        for scene in scenes[:10]:  # أول 10 مشاهد
            for turn in scene.dialogue[:3]:  # أول 3 حوارات لكل مشهد
                sample_dialogues.append({
                    "مشهد": scene.scene_number,
                    "متحدث": turn.speaker,
                    "نص": turn.text[:200]  # أول 200 حرف
                })
        
        prompt = f"""أنت محلل نفسي ومتخصص في تحليل المشاعر في النصوص الدرامية.

حلل المشاعر في هذه العينة من الحوارات:

{json.dumps(sample_dialogues, ensure_ascii=False, indent=2)}

قدم تحليلاً يشمل:
1. الحالة العاطفية العامة للسيناريو
2. تطور المشاعر عبر المشاهد
3. الشخصيات الأكثر عاطفية
4. أنماط المشاعر المتكررة

أجب بصيغة JSON:
{{
    "الحالة_العامة": "...",
    "تطور_المشاعر": [...],
    "الشخصيات_العاطفية": [...],
    "الأنماط_المتكررة": [...]
}}"""
        
        result = self._call_gemini(prompt)
        try:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"raw_analysis": result}
    
    def analyze_character_development(self, scenes: List[Scene]) -> dict:
        """تحليل تطور الشخصيات"""
        if not self.client:
            return {}
        
        logger.info("👥 بدء تحليل تطور الشخصيات بواسطة Gemini...")
        
        # تجميع الشخصيات الرئيسية
        char_counts = defaultdict(int)
        for scene in scenes:
            for turn in scene.dialogue:
                char_counts[turn.speaker] += 1
        
        main_chars = [char for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        analyses = {}
        for char in main_chars:
            # جمع حوارات الشخصية من البداية والوسط والنهاية
            char_dialogues = []
            for scene in scenes:
                for turn in scene.dialogue:
                    if turn.speaker == char:
                        char_dialogues.append({
                            "مشهد": scene.scene_number,
                            "نص": turn.text[:150]
                        })
            
            if len(char_dialogues) < 3: continue
            
            # تقسيم الحوارات إلى ثلاث مراحل
            third = len(char_dialogues) // 3
            start = char_dialogues[:third]
            middle = char_dialogues[third:2*third]
            end = char_dialogues[2*third:]
            
            prompt = f"""أنت ناقد أدبي متخصص في تحليل الشخصيات الدرامية.

حلل تطور شخصية "{char}" عبر السيناريو:

حوارات البداية:
{json.dumps(start, ensure_ascii=False, indent=2)}

حوارات الوسط:
{json.dumps(middle, ensure_ascii=False, indent=2)}

حوارات النهاية:
{json.dumps(end, ensure_ascii=False, indent=2)}

قدم تحليلاً يشمل:
1. السمات الشخصية الأولية
2. نقاط التحول الدرامية
3. التطور النفسي والعاطفي
4. العلاقة مع الشخصيات الأخرى
5. القوس الدرامي للشخصية (Character Arc)

أجب بصيغة JSON:
{{
    "الشخصية": "{char}",
    "السمات_الأولية": [...],
    "نقاط_التحول": [...],
    "التطور_النفسي": "...",
    "القوس_الدرامي": "...",
    "الدور_في_القصة": "..."
}}"""
            
            result = self._call_gemini(prompt)
            try:
                json_match = re.search(r'\{[\s\S]*\}', result)
                if json_match:
                    analyses[char] = json.loads(json_match.group())
            except:
                analyses[char] = {"raw_analysis": result}
            
            time.sleep(1)  # تجنب rate limiting
        
        return analyses
    
    def analyze_plot(self, scenes: List[Scene]) -> dict:
        """تحليل الحبكة الدرامية"""
        if not self.client:
            return {}
        
        logger.info("📖 بدء تحليل الحبكة الدرامية بواسطة Gemini...")
        
        # تجميع ملخص المشاهد
        scene_summaries = []
        for scene in scenes:
            summary = {
                "رقم": scene.scene_number,
                "المكان": scene.location,
                "الوقت": scene.time_of_day,
                "الشخصيات": scene.characters[:5],
                "عدد_الحوارات": len(scene.dialogue),
                "نص_مختصر": scene.full_text[:300] if scene.full_text else ""
            }
            scene_summaries.append(summary)
        
        prompt = f"""أنت محلل سيناريو محترف ومتخصص في البنية الدرامية.

حلل الحبكة الدرامية للسيناريو التالي:

عدد المشاهد: {len(scenes)}
ملخص المشاهد:
{json.dumps(scene_summaries[:15], ensure_ascii=False, indent=2)}

قدم تحليلاً شاملاً يتضمن:

1. **البنية الدرامية** (Three-Act Structure):
   - التمهيد (Setup)
   - المواجهة (Confrontation)  
   - الحل (Resolution)

2. **عناصر الحبكة**:
   - الصراع الرئيسي
   - الصراعات الفرعية
   - نقطة التحول الأولى
   - الذروة (Climax)
   - الحل

3. **الثيمات والموضوعات** الرئيسية

4. **الإيقاع الدرامي** (Pacing)

5. **نقاط القوة والضعف** في الحبكة

أجب بصيغة JSON:
{{
    "البنية_الدرامية": {{
        "التمهيد": "...",
        "المواجهة": "...",
        "الحل": "..."
    }},
    "الصراع_الرئيسي": "...",
    "الصراعات_الفرعية": [...],
    "نقاط_التحول": [...],
    "الذروة": "...",
    "الثيمات": [...],
    "الإيقاع": "...",
    "نقاط_القوة": [...],
    "نقاط_الضعف": [...],
    "التقييم_العام": "..."
}}"""
        
        result = self._call_gemini(prompt)
        try:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"raw_analysis": result}
    
    def generate_screenplay_report(self, scenes: List[Scene], output_dir: Path) -> dict:
        """توليد تقرير شامل عن السيناريو"""
        if not self.client:
            logger.warning("Gemini غير متوفر - تخطي التقرير المتقدم")
            return {}
        
        logger.info("📊 توليد التقرير الشامل...")
        
        report = {
            "معلومات_عامة": {
                "عدد_المشاهد": len(scenes),
                "عدد_الشخصيات": len(set(c for s in scenes for c in s.characters)),
                "إجمالي_الحوارات": sum(len(s.dialogue) for s in scenes)
            },
            "تحليل_المشاعر": self.analyze_sentiment_deep(scenes),
            "تطور_الشخصيات": self.analyze_character_development(scenes),
            "تحليل_الحبكة": self.analyze_plot(scenes)
        }
        
        # حفظ التقرير
        report_path = output_dir / "gemini_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ تم حفظ تقرير Gemini: {report_path}")
        return report

# ---------------------------------------------------------
# 8. طبقة التصدير والإنتاج (Production Exporter)
# ---------------------------------------------------------
class DatasetExporter:
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
        تهيئة مصدّر مجموعات البيانات المحسّن

        Args:
            output_dir: مجلد المخرجات
            min_words: الحد الأدنى لعدد الكلمات (لفلترة الجودة)
            sentiment_threshold: عتبة المشاعر العالية للاحتفاظ بالحوارات القصيرة
            min_action_length: الحد الأدنى لطول السطر الوصفي المهم
            apply_quality_filter: تطبيق فلترة الجودة
            apply_context_enrichment: تطبيق إثراء السياق
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.apply_quality_filter = apply_quality_filter and AL_RAWI_V4_AVAILABLE
        self.apply_context_enrichment = apply_context_enrichment and AL_RAWI_V4_AVAILABLE

        # تهيئة وحدات الراوي 4.0
        self.quality_filter = None
        self.context_enricher = None

        if AL_RAWI_V4_AVAILABLE:
            if apply_quality_filter:
                self.quality_filter = QualityFilter(min_words, sentiment_threshold)
                logger.info(f"تم تهيئة فلترة الجودة: الحد الأدنى للكلمات={min_words}")
            if apply_context_enrichment:
                self.context_enricher = ContextEnricher(min_action_length)
                logger.info(f"تم تهيئة إثراء السياق: الحد الأدنى للسطر الوصفي={min_action_length}")
        else:
            logger.warning("وحدات الراوي 4.0 غير متوفرة - استخدام الوضع الأساسي")

    def _filter_scenes_quality(self, scenes: List[Scene]) -> List[Scene]:
        """
        تطبيق فلترة الجودة على المشاهد

        يزيل الحوارات القصيرة (أقل من 3 كلمات) إلا إذا كانت
        ذات مشاعر عالية (درجة > 0.8)

        Args:
            scenes: قائمة المشاهد

        Returns:
            المشاهد مع الحوارات المفلترة
        """
        if not self.quality_filter:
            return scenes

        filtered_scenes = []
        for scene in scenes:
            filtered_dialogue = []
            for turn in scene.dialogue:
                # التحقق من عدد الكلمات
                word_count = count_arabic_words(turn.text)

                # قاعدة 1: الحوارات الطويلة تُحفظ دائماً
                if word_count >= 3:
                    filtered_dialogue.append(turn)
                # قاعدة 2: الحوارات القصيرة ذات المشاعر القوية تُحفظ
                elif turn.sentiment_score >= 0.8:
                    filtered_dialogue.append(turn)
                    logger.debug(f"الاحتفاظ بحوار قصير عاطفي: '{turn.text[:30]}...'")
                # قاعدة 3: إذا لم يكن تحليل المشاعر متوفراً، احتفظ بالحوار
                elif turn.sentiment == "unknown":
                    filtered_dialogue.append(turn)
                else:
                    logger.debug(f"حذف حوار قصير: '{turn.text[:30]}...' ({word_count} كلمات)")

            # إنشاء نسخة من المشهد مع الحوارات المفلترة
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
                characters=[t.speaker for t in filtered_dialogue if t.speaker],
                full_text=scene.full_text,
                embedding=scene.embedding
            )
            filtered_scenes.append(filtered_scene)

        original_count = sum(len(s.dialogue) for s in scenes)
        filtered_count = sum(len(s.dialogue) for s in filtered_scenes)
        logger.info(f"فلترة الجودة: {original_count} → {filtered_count} حوار (حذف {original_count - filtered_count})")

        return filtered_scenes

    def _get_last_significant_action(self, actions: List[str], min_length: int = 10) -> str:
        """
        استخراج آخر سطر وصفي مهم من قائمة الأفعال

        Args:
            actions: قائمة الأسطر الوصفية
            min_length: الحد الأدنى لطول السطر المهم

        Returns:
            آخر سطر وصفي مهم، أو سلسلة فارغة
        """
        if not actions:
            return ""

        transitions = {"قطع", "كات", "CUT", "CUT TO", "FADE OUT", "FADE IN", "DISSOLVE"}

        for action in reversed(actions):
            clean = action.strip()
            # استبعاد الانتقالات والأسطر القصيرة
            if clean.upper() in transitions or clean in transitions:
                continue
            if len(clean) < min_length:
                continue
            return clean

        return ""

    def _build_enriched_scene_setup(self, scene: Scene, last_action: str) -> str:
        """
        بناء وصف المشهد المحسّن مع السياق الوصفي

        Args:
            scene: كائن المشهد
            last_action: آخر سطر وصفي مهم

        Returns:
            وصف المشهد المحسّن
        """
        parts = []

        if scene.heading:
            parts.append(scene.heading)

        if scene.location:
            location_info = f"المكان: {scene.location}"
            if scene.int_ext:
                location_info += f" ({scene.int_ext})"
            parts.append(location_info)

        if scene.time_of_day:
            parts.append(f"الوقت: {scene.time_of_day}")

        if scene.time_period and scene.time_period != "غير محدد":
            parts.append(f"الفترة: {scene.time_period}")

        if last_action:
            parts.append(f"[سياق: {last_action}]")

        return "\n".join(parts) if parts else ""

    def export_contextual_alpaca(self, scenes: List[Scene]):
        """
        تصدير بصيغة Alpaca مع نافذة سياق (Sliding Window) وإثراء السياق.

        التحسينات في الإصدار 4.0:
        - فلترة الحوارات منخفضة الجودة
        - إضافة السياق الوصفي من الأسطر الوصفية
        - تضمين معلومات الفترة الزمنية
        """
        # تطبيق فلترة الجودة أولاً
        if self.apply_quality_filter:
            scenes = self._filter_scenes_quality(scenes)

        data = []
        enrichment_stats = {"scenes_processed": 0, "dialogues_enriched": 0, "actions_used": 0}

        for scene in scenes:
            dialogue = scene.dialogue
            if not dialogue: continue

            enrichment_stats["scenes_processed"] += 1

            # نافذة السياق (قائمة انتظار)
            context_buffer = []

            # استخراج آخر سطر وصفي مهم (إثراء السياق)
            last_action = ""
            if self.apply_context_enrichment:
                last_action = self._get_last_significant_action(scene.actions)
                if last_action:
                    enrichment_stats["actions_used"] += 1

            # بناء وصف المشهد المحسّن
            if self.apply_context_enrichment:
                scene_setup = self._build_enriched_scene_setup(scene, last_action)
            else:
                scene_setup = f"المشهد: {scene.heading}\nالمكان: {scene.location}\nالوقت: {scene.time_of_day}"

            for i, turn in enumerate(dialogue):
                enrichment_stats["dialogues_enriched"] += 1

                # إذا لم يكن هناك سياق سابق، نستخدم وصف المشهد
                current_history = "\n".join(context_buffer) if context_buffer else "بداية الحوار."

                full_input = f"{scene_setup}\n\nسياق الحديث السابق:\n{current_history}\n\nالمتحدث: {turn.speaker}"

                if turn.sentiment != "unknown":
                    full_input += f" (الحالة الشعورية: {turn.sentiment})"

                entry = {
                    "instruction": "أكمل الحوار التالي بناءً على السياق المعطى",
                    "input": full_input,
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
                data.append(entry)

                # تحديث النافذة: نضيف الرد الحالي
                context_buffer.append(f"{turn.speaker}: {turn.text}")
                # نحذف القديم إذا تجاوزنا الحد المسموح
                if len(context_buffer) > Config.CONTEXT_WINDOW_SIZE:
                    context_buffer.pop(0)

        if self.apply_context_enrichment:
            logger.info(
                f"إثراء السياق: {enrichment_stats['dialogues_enriched']} حوار "
                f"من {enrichment_stats['scenes_processed']} مشهد "
                f"باستخدام {enrichment_stats['actions_used']} سطر وصفي"
            )

        self._write_json(data, "train_alpaca_contextual.json")

    def export_sharegpt(self, scenes: List[Scene]):
        """تصدير بصيغة ShareGPT (للنماذج التي تدعم المحادثات الطويلة)"""
        # تطبيق فلترة الجودة أولاً
        if self.apply_quality_filter:
            scenes = self._filter_scenes_quality(scenes)

        data = []
        for scene in scenes:
            if not scene.dialogue: continue
            
            conversations = [{
                "from": "system",
                "value": f"هذا مشهد تمثيلي يدور في {scene.location} ({scene.time_of_day}). تقمص أدوار الشخصيات بدقة."
            }]
            
            for turn in scene.dialogue:
                conversations.append({
                    "from": "user",
                    "value": f"[{turn.speaker}]: {turn.text}"
                })
            
            data.append({"conversations": conversations})
        
        self._write_json(data, "train_sharegpt.json")

    def export_rag_jsonl(self, scenes: List[Scene]):
        """تصدير قاعدة بيانات كاملة للبحث (RAG)"""
        # تطبيق فلترة الجودة أولاً
        if self.apply_quality_filter:
            scenes = self._filter_scenes_quality(scenes)

        data = [asdict(s) for s in scenes]
        with open(self.output_dir / "rag_dataset.jsonl", 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logger.info("تم تصدير ملفات RAG.")

    def export_stats(self, graph):
        """Export character statistics to CSV"""
        if graph is None or not NETWORKX_AVAILABLE:
            logger.warning("تخطي تصدير الإحصائيات - الشبكة غير متوفرة")
            return
        stats = []
        for node in graph.nodes():
            stats.append({
                "character": node,
                "interactions": graph.degree(node),
                "centrality": nx.degree_centrality(graph)[node] if len(graph) > 0 else 0
            })
        stats = sorted(stats, key=lambda x: x['interactions'], reverse=True)
        pd.DataFrame(stats).to_csv(self.output_dir / "character_stats.csv", index=False)
        logger.info(f"تم تصدير إحصائيات الشخصيات ({len(stats)} شخصية)")

    def export_dialogue_csv(self, scenes: List[Scene]):
        """تصدير الحوارات بصيغة CSV"""
        # تطبيق فلترة الجودة أولاً
        if self.apply_quality_filter:
            scenes = self._filter_scenes_quality(scenes)

        rows = []
        for scene in scenes:
            for turn in scene.dialogue:
                rows.append({
                    "scene_id": scene.scene_id,
                    "scene_number": scene.scene_number,
                    "location": scene.location,
                    "time_of_day": scene.time_of_day,
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "normalized_text": turn.normalized_text,
                    "sentiment": turn.sentiment,
                    "sentiment_score": turn.sentiment_score,
                    "word_count": count_arabic_words(turn.text)
                })
        if rows:
            pd.DataFrame(rows).to_csv(self.output_dir / "dialogue_turns.csv", index=False, encoding='utf-8-sig')
            logger.info(f"تم تصدير الحوارات ({len(rows)} جملة)")

    def export_summary(self, scenes: List[Scene]):
        """تصدير ملخص السيناريو"""
        total_dialogue = sum(len(s.dialogue) for s in scenes)
        total_characters = len(set(c for s in scenes for c in s.characters))
        total_words = sum(count_arabic_words(s.full_text) for s in scenes)
        
        summary = {
            "عدد_المشاهد": len(scenes),
            "عدد_الحوارات": total_dialogue,
            "عدد_الشخصيات": total_characters,
            "عدد_الكلمات_الإجمالي": total_words,
            "متوسط_الحوارات_لكل_مشهد": round(total_dialogue / len(scenes), 2) if scenes else 0,
            "الشخصيات": list(set(c for s in scenes for c in s.characters))
        }
        
        with open(self.output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("تم تصدير ملخص السيناريو")

    def get_enhancement_stats(self) -> dict:
        """
        الحصول على إحصائيات التحسينات المطبقة

        Returns:
            قاموس بإحصائيات فلترة الجودة وإثراء السياق
        """
        stats = {
            "quality_filter_enabled": self.apply_quality_filter,
            "context_enrichment_enabled": self.apply_context_enrichment,
            "al_rawi_v4_available": AL_RAWI_V4_AVAILABLE
        }

        if self.quality_filter:
            stats["quality_filter_stats"] = self.quality_filter.get_filter_stats()
        if self.context_enricher:
            stats["context_enricher_stats"] = self.context_enricher.get_enrichment_stats()

        return stats

    def _write_json(self, data: Any, filename: str):
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"تم تصدير: {filename} ({len(data)} عينة)")

# ---------------------------------------------------------
# 9. التنفيذ الرئيسي (Main Orchestrator)
# ---------------------------------------------------------
def main(input_path: str, output_folder: str = "alrawi_output"):
    print(f"\n{'='*60}")
    print(f"نظام الراوي الإصدار 4.0 - معالجة السيناريوهات العربية")
    print(f"{'='*60}")
    print(f"الملف المدخل: {input_path}")
    start_global = time.time()

    # عرض حالة API Key
    if UNSTRUCTURED_API_KEY:
        logger.info("✅ تم تحميل UNSTRUCTURED_API_KEY من ملف .env")
    else:
        logger.warning("⚠️ UNSTRUCTURED_API_KEY غير موجود في ملف .env")

    # عرض حالة وحدات الراوي 4.0
    if AL_RAWI_V4_AVAILABLE:
        logger.info("✅ وحدات الراوي 4.0 متوفرة (إثراء السياق + فلترة الجودة)")
    else:
        logger.warning("⚠️ وحدات الراوي 4.0 غير متوفرة - استخدام الوضع الأساسي")

    # 1. القراءة (Ingestion) - اختيار المعالج حسب نوع الملف
    ingestor = get_ingestor(input_path)
    raw_lines = ingestor.process(input_path)

    if not raw_lines:
        print("فشلت عملية استخراج النص.")
        return

    # 2. التحليل (Parsing)
    parser = ScreenplayParser()
    scenes = parser.parse(raw_lines)
    print(f"✅ تم استخراج {len(scenes)} مشهد.")

    # 3. الإثراء (Enrichment)
    social_graph = None
    if ML_AVAILABLE:
        enricher = AIEnricher(use_gpu=True) # اجعلها False إذا لم يوجد GPU
        enricher.enrich(scenes)
        social_graph = enricher.build_social_graph(scenes)

    # 4. التحليل المتقدم بـ Gemini
    # إنشاء مجلد المخرجات
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        gemini_analyzer = GeminiAnalyzer()
        gemini_analyzer.generate_screenplay_report(scenes, Path(output_folder))

    # 5. التصدير (Export) - مع تحسينات الراوي 4.0
    exporter = DatasetExporter(
        output_folder,
        min_words=3,              # الحد الأدنى لعدد الكلمات
        sentiment_threshold=0.8,  # عتبة المشاعر العالية
        min_action_length=10,     # الحد الأدنى لطول السطر الوصفي
        apply_quality_filter=True,      # تفعيل فلترة الجودة
        apply_context_enrichment=True   # تفعيل إثراء السياق
    )
    exporter.export_contextual_alpaca(scenes) # Alpaca المطور مع إثراء السياق
    exporter.export_sharegpt(scenes)          # ShareGPT
    exporter.export_rag_jsonl(scenes)         # Vector DB
    exporter.export_dialogue_csv(scenes)      # CSV للحوارات
    exporter.export_summary(scenes)           # ملخص السيناريو
    if ML_AVAILABLE and social_graph:
        exporter.export_stats(social_graph)

    # عرض إحصائيات التحسينات
    enhancement_stats = exporter.get_enhancement_stats()
    print(f"\n{'='*60}")
    print("إحصائيات التحسينات (الراوي 4.0)")
    print(f"{'='*60}")
    print(f"فلترة الجودة: {'مفعّلة' if enhancement_stats['quality_filter_enabled'] else 'معطّلة'}")
    print(f"إثراء السياق: {'مفعّل' if enhancement_stats['context_enrichment_enabled'] else 'معطّل'}")

    print(f"\n✅ تمت المهمة بنجاح في {time.time() - start_global:.2f} ثانية.")
    print(f"📂 المخرجات في المجلد: {output_folder}")

if __name__ == "__main__":
    # ===============================================
    # ⚙️ إعدادات الملفات - عدّل هنا مباشرة
    # ===============================================
    DEFAULT_INPUT_FILE = r"E:\PREPA\Extracted_Dataset\1.txt"
    DEFAULT_OUTPUT_DIR = "dataset_output"
    # ===============================================
    
    import argparse
    parser = argparse.ArgumentParser(description="معالج السيناريوهات المتقدم")
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="مسار ملف السيناريو")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="مجلد المخرجات")
    args = parser.parse_args()
    
    main(args.input, args.out)
