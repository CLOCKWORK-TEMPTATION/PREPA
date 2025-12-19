import os
import re
import json
import math
import mimetypes
import hashlib
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

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

    def start_scene(meta: dict[str, Any]):
        nonlocal current, turn_counter, pending_location
        if current:
            flush_turn()
            finalize_scene(current)
            scenes.append(current)
        sid = f"S{meta.get('scene_number', len(scenes) + 1):04d}"
        current = Scene(
            scene_id=sid,
            scene_number=meta.get("scene_number"),
            heading=meta.get("heading"),
            location=None,
            time_of_day=meta.get("time_of_day"),
            int_ext=meta.get("int_ext"),
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
            }
        )
        for dt in sc.dialogue:
            dialogue_rows.append(
                {
                    "scene_id": dt.scene_id,
                    "turn_id": dt.turn_id,
                    "speaker": dt.speaker,
                    "text": dt.text,
                    "element_ids": dt.element_ids,
                    "word_count": len((dt.text or "").split()),
                }
            )

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
