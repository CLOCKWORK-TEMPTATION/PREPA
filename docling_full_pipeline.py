"""
سكريبت معالجة السيناريو العربي باستخدام Docling بكامل قدراته
يستخدم: Layout Analysis, TableFormer, OCR (العربية), DoclingDocument
"""

import json
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import Counter

# استيراد مكونات Docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    EasyOcrOptions,
)
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

# إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('docling_full_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedScene:
    """فئة تمثل مشهد واحد مستخرج"""
    scene_number: int
    time_of_day: str
    location_type: str
    location_name: str
    dialogues: List[Dict[str, str]] = field(default_factory=list)
    characters: List[str] = field(default_factory=list)
    stage_directions: List[str] = field(default_factory=list)
    raw_content: str = ""


@dataclass
class DocumentAnalysis:
    """تحليل شامل للوثيقة"""
    title: str = ""
    author: str = ""
    director: str = ""
    series_name: str = ""
    year: str = ""
    total_pages: int = 0
    total_scenes: int = 0
    total_dialogues: int = 0
    total_characters: int = 0
    total_tables: int = 0
    total_pictures: int = 0
    processing_time: float = 0.0


class DoclingFullPipeline:
    """
    معالج السيناريو باستخدام Docling بكامل قدراته
    
    يستخدم:
    - Layout Analysis: تحليل التخطيط البصري للكشف عن العناوين والنصوص
    - TableFormer: استخراج الجداول بدقة عالية
    - OCR (EasyOCR): قراءة النص العربي
    - DoclingDocument: البنية الموحدة للعناصر
    """
    
    def __init__(self, input_file: str, output_dir: str = "docling_output"):
        """
        تهيئة المعالج
        
        Args:
            input_file: مسار ملف PDF المدخل
            output_dir: مجلد المخرجات
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.docling_result = None
        self.document = None
        self.scenes: List[ExtractedScene] = []
        self.analysis = DocumentAnalysis()
        self.character_stats: Dict[str, int] = {}
        self.body_items_by_type: Dict[str, List[Any]] = {}
        
        logger.info(f"تم تهيئة المعالج للملف: {self.input_file.name}")
    
    def _create_pipeline_options(self) -> PdfPipelineOptions:
        """
        إنشاء خيارات خط الأنابيب مع تفعيل جميع القدرات
        
        Returns:
            كائن PdfPipelineOptions مُهيأ بالكامل
        """
        pipeline_options = PdfPipelineOptions()
        
        # تفعيل OCR للنص العربي
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions(
            lang=["ar", "en"],  # دعم العربية والإنجليزية
            force_full_page_ocr=False,  # Lazy OCR - يعمل عند الحاجة فقط
        )
        
        # تفعيل تحليل بنية الجداول (TableFormer)
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # إعدادات المسرّع (GPU إذا متاح، وإلا CPU)
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.AUTO
        )
        
        logger.info("تم تهيئة خيارات خط الأنابيب:")
        logger.info("  - OCR: مُفعّل (العربية + الإنجليزية)")
        logger.info("  - TableFormer: ACCURATE mode")
        logger.info("  - Accelerator: AUTO")
        
        return pipeline_options
    
    def _create_document_converter(self) -> DocumentConverter:
        """
        إنشاء محول الوثائق مع الخيارات المتقدمة
        
        Returns:
            كائن DocumentConverter مُهيأ
        """
        pipeline_options = self._create_pipeline_options()
        
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        return doc_converter
    
    def process_document(self):
        """
        معالجة الوثيقة باستخدام Docling
        """
        logger.info("=" * 60)
        logger.info("بدء معالجة الوثيقة باستخدام Docling")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # إنشاء المحول
            converter = self._create_document_converter()
            
            # تحويل الوثيقة
            logger.info(f"جاري تحويل: {self.input_file}")
            self.docling_result = converter.convert(str(self.input_file))
            self.document = self.docling_result.document
            
            end_time = time.time()
            self.analysis.processing_time = end_time - start_time
            
            logger.info(f"تم التحويل في {self.analysis.processing_time:.2f} ثانية")
            
        except Exception as e:
            logger.error(f"خطأ في معالجة الوثيقة: {e}")
            raise
    
    def analyze_document_structure(self):
        """
        تحليل بنية الوثيقة واستخراج المعلومات
        """
        logger.info("جاري تحليل بنية الوثيقة...")
        
        doc = self.document
        
        # معلومات أساسية
        self.analysis.total_pages = len(doc.pages) if hasattr(doc, 'pages') else 0
        
        # تصنيف العناصر حسب النوع
        self.body_items_by_type = {
            'texts': [],
            'section_headers': [],
            'tables': [],
            'pictures': [],
            'page_headers': [],
            'page_footers': [],
            'captions': [],
            'footnotes': [],
            'lists': [],
            'other': []
        }
        
        # استخراج العناصر من body_items
        if hasattr(doc, 'body'):
            self._process_body_items(doc.body)
        
        # إحصائيات
        self.analysis.total_tables = len(self.body_items_by_type['tables'])
        self.analysis.total_pictures = len(self.body_items_by_type['pictures'])
        
        logger.info(f"تم استخراج:")
        logger.info(f"  - عناصر نصية: {len(self.body_items_by_type['texts'])}")
        logger.info(f"  - عناوين أقسام: {len(self.body_items_by_type['section_headers'])}")
        logger.info(f"  - جداول: {self.analysis.total_tables}")
        logger.info(f"  - صور: {self.analysis.total_pictures}")
    
    def _process_body_items(self, body):
        """
        معالجة عناصر الجسم وتصنيفها
        
        Args:
            body: جسم الوثيقة
        """
        # محاولة الوصول للعناصر بطرق مختلفة
        items = []
        
        if hasattr(body, 'children'):
            items = body.children
        elif hasattr(body, 'items'):
            items = body.items
        elif isinstance(body, list):
            items = body
        
        for item in items:
            item_type = self._get_item_type(item)
            
            if item_type in self.body_items_by_type:
                self.body_items_by_type[item_type].append(item)
            else:
                self.body_items_by_type['other'].append(item)
            
            # معالجة العناصر المتداخلة
            if hasattr(item, 'children'):
                self._process_body_items(item)
    
    def _get_item_type(self, item) -> str:
        """
        تحديد نوع العنصر
        
        Args:
            item: العنصر المراد تحديد نوعه
            
        Returns:
            نوع العنصر كنص
        """
        type_name = type(item).__name__.lower()
        
        type_mapping = {
            'textitem': 'texts',
            'text': 'texts',
            'paragraph': 'texts',
            'sectionheader': 'section_headers',
            'sectionheaderitem': 'section_headers',
            'header': 'section_headers',
            'table': 'tables',
            'tableitem': 'tables',
            'picture': 'pictures',
            'pictureitem': 'pictures',
            'image': 'pictures',
            'pageheader': 'page_headers',
            'pagefooter': 'page_footers',
            'caption': 'captions',
            'footnote': 'footnotes',
            'list': 'lists',
            'listitem': 'lists',
        }
        
        for key, value in type_mapping.items():
            if key in type_name:
                return value
        
        return 'other'
    
    def extract_metadata(self):
        """
        استخراج المعلومات الوصفية من الوثيقة
        """
        logger.info("جاري استخراج المعلومات الوصفية...")
        
        # استخدام تصدير Markdown للحصول على النص الكامل
        markdown_text = self.document.export_to_markdown()
        lines = markdown_text.split('\n')[:30]
        
        for line in lines:
            line = line.strip()
            
            # البحث عن العنوان
            if line.startswith('#') and not self.analysis.title:
                self.analysis.title = line.lstrip('#').strip()
            
            # البحث عن المؤلف
            if 'تأليف' in line or 'المؤلف' in line:
                self.analysis.author = line.replace('تأليف', '').replace('المؤلف', '').strip()
            
            # البحث عن المخرج
            if 'إخراج' in line or 'المخرج' in line:
                self.analysis.director = line.replace('إخراج', '').replace('المخرج', '').strip()
            
            # البحث عن اسم المسلسل
            if 'الشعلة' in line:
                self.analysis.series_name = line.strip()
            
            # البحث عن السنة
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', line)
            if year_match and not self.analysis.year:
                self.analysis.year = year_match.group()
        
        logger.info(f"العنوان: {self.analysis.title}")
        logger.info(f"المسلسل: {self.analysis.series_name}")
        logger.info(f"السنة: {self.analysis.year}")
    
    def extract_scenes_and_dialogues(self):
        """
        استخراج المشاهد والحوارات من النص
        """
        logger.info("جاري استخراج المشاهد والحوارات...")
        
        import re
        
        # الحصول على النص الكامل
        markdown_text = self.document.export_to_markdown()
        lines = markdown_text.split('\n')
        
        # أنماط للكشف
        scene_pattern = re.compile(r'مشهد\s*(\d+)\s*(.*?)\s*(ليل|نهار)[\s\-]*(داخلي|خارجي)?', re.UNICODE)
        dialogue_pattern = re.compile(r'^([^:]+)\s*:\s*(.*)$', re.UNICODE)
        
        current_scene = None
        scene_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # التحقق من مشهد جديد
            scene_match = scene_pattern.search(line)
            if scene_match:
                # حفظ المشهد السابق
                if current_scene:
                    current_scene.raw_content = '\n'.join(scene_content)
                    self._analyze_scene_content(current_scene, scene_content)
                    self.scenes.append(current_scene)
                
                # بدء مشهد جديد
                scene_number = int(scene_match.group(1))
                location_info = scene_match.group(2).strip() if scene_match.group(2) else ""
                time_of_day = scene_match.group(3) if scene_match.group(3) else ""
                location_type = scene_match.group(4) if scene_match.group(4) else ""
                
                current_scene = ExtractedScene(
                    scene_number=scene_number,
                    time_of_day=time_of_day,
                    location_type=location_type,
                    location_name=location_info
                )
                scene_content = []
                continue
            
            # جمع محتوى المشهد
            if current_scene and line != 'قطع':
                scene_content.append(line)
        
        # حفظ المشهد الأخير
        if current_scene:
            current_scene.raw_content = '\n'.join(scene_content)
            self._analyze_scene_content(current_scene, scene_content)
            self.scenes.append(current_scene)
        
        # تحديث الإحصائيات
        self.analysis.total_scenes = len(self.scenes)
        self.analysis.total_dialogues = sum(len(s.dialogues) for s in self.scenes)
        self.analysis.total_characters = len(self.character_stats)
        
        logger.info(f"تم استخراج {len(self.scenes)} مشهد")
        logger.info(f"إجمالي الحوارات: {self.analysis.total_dialogues}")
        logger.info(f"عدد الشخصيات: {len(self.character_stats)}")
    
    def _analyze_scene_content(self, scene: ExtractedScene, content_lines: List[str]):
        """
        تحليل محتوى المشهد لاستخراج الحوارات
        
        Args:
            scene: كائن المشهد
            content_lines: أسطر المحتوى
        """
        import re
        
        dialogue_pattern = re.compile(r'^([^:]+)\s*:\s*(.*)$', re.UNICODE)
        stage_keywords = ['يجلس', 'يقف', 'يدخل', 'يخرج', 'ينظر', 'يتحدث',
                         'تجلس', 'تقف', 'تدخل', 'تخرج', 'تنظر', 'تتحدث',
                         'ينهض', 'تنهض', 'يمشي', 'تمشي', 'بينما', 'فجأة']
        
        current_character = None
        current_dialogue = []
        
        for line in content_lines:
            line = line.strip()
            if not line:
                continue
            
            # التحقق من حوار
            match = dialogue_pattern.match(line)
            if match:
                potential_char = match.group(1).strip()
                dialogue_text = match.group(2).strip()
                
                # التحقق من أنه ليس إرشاد مسرحي
                is_stage = any(kw in line for kw in stage_keywords)
                
                if not is_stage and len(potential_char.split()) <= 3:
                    # حفظ الحوار السابق
                    if current_character and current_dialogue:
                        scene.dialogues.append({
                            'character': current_character,
                            'text': ' '.join(current_dialogue)
                        })
                        if current_character not in scene.characters:
                            scene.characters.append(current_character)
                        self.character_stats[current_character] = self.character_stats.get(current_character, 0) + 1
                    
                    current_character = potential_char
                    current_dialogue = [dialogue_text] if dialogue_text else []
                    continue
            
            # التحقق من إرشاد مسرحي
            is_stage = any(kw in line for kw in stage_keywords)
            if is_stage:
                if current_character and current_dialogue:
                    scene.dialogues.append({
                        'character': current_character,
                        'text': ' '.join(current_dialogue)
                    })
                    if current_character not in scene.characters:
                        scene.characters.append(current_character)
                    self.character_stats[current_character] = self.character_stats.get(current_character, 0) + 1
                    current_character = None
                    current_dialogue = []
                scene.stage_directions.append(line)
            elif current_character:
                current_dialogue.append(line)
        
        # حفظ الحوار الأخير
        if current_character and current_dialogue:
            scene.dialogues.append({
                'character': current_character,
                'text': ' '.join(current_dialogue)
            })
            if current_character not in scene.characters:
                scene.characters.append(current_character)
            self.character_stats[current_character] = self.character_stats.get(current_character, 0) + 1
    
    def export_all_formats(self) -> Dict[str, Path]:
        """
        تصدير البيانات بجميع التنسيقات المتاحة
        
        Returns:
            قاموس بمسارات الملفات المُصدَّرة
        """
        logger.info("جاري تصدير البيانات...")
        
        output_files = {}
        stem = self.input_file.stem
        
        # 1. تصدير JSON الكامل من Docling
        json_docling_file = self.output_dir / f"{stem}_docling_raw.json"
        with open(json_docling_file, 'w', encoding='utf-8') as f:
            json.dump(self.document.export_to_dict(), f, ensure_ascii=False, indent=2)
        output_files['json_docling_raw'] = json_docling_file
        logger.info(f"تم تصدير: {json_docling_file.name}")
        
        # 2. تصدير Dataset المنظم
        json_dataset_file = self.output_dir / f"{stem}_dataset.json"
        dataset = {
            'metadata': asdict(self.analysis),
            'docling_info': {
                'library_version': 'v2.65.0',
                'pipeline': 'PDF + OCR (ar/en) + TableFormer (ACCURATE)',
                'processing_time_seconds': self.analysis.processing_time
            },
            'statistics': {
                'body_items': {k: len(v) for k, v in self.body_items_by_type.items()},
                'character_appearances': self.character_stats
            },
            'scenes': [
                {
                    'scene_number': s.scene_number,
                    'time_of_day': s.time_of_day,
                    'location_type': s.location_type,
                    'location_name': s.location_name,
                    'dialogues_count': len(s.dialogues),
                    'dialogues': s.dialogues,
                    'characters': s.characters,
                    'stage_directions': s.stage_directions
                }
                for s in self.scenes
            ]
        }
        with open(json_dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        output_files['json_dataset'] = json_dataset_file
        logger.info(f"تم تصدير: {json_dataset_file.name}")
        
        # 3. تصدير CSV للحوارات
        csv_file = self.output_dir / f"{stem}_dialogues.csv"
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['رقم_المشهد', 'الوقت', 'المكان_نوع', 'المكان_اسم', 'الشخصية', 'الحوار', 'عدد_الكلمات'])
            for scene in self.scenes:
                for dialogue in scene.dialogues:
                    writer.writerow([
                        scene.scene_number,
                        scene.time_of_day,
                        scene.location_type,
                        scene.location_name,
                        dialogue['character'],
                        dialogue['text'],
                        len(dialogue['text'].split())
                    ])
        output_files['csv_dialogues'] = csv_file
        logger.info(f"تم تصدير: {csv_file.name}")
        
        # 4. تصدير Markdown
        md_file = self.output_dir / f"{stem}_markdown.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(self.document.export_to_markdown())
        output_files['markdown'] = md_file
        logger.info(f"تم تصدير: {md_file.name}")
        
        # 5. تصدير DocTags (للتدريب)
        doctags_file = self.output_dir / f"{stem}_doctags.txt"
        with open(doctags_file, 'w', encoding='utf-8') as f:
            f.write(self.document.export_to_doctags())
        output_files['doctags'] = doctags_file
        logger.info(f"تم تصدير: {doctags_file.name}")
        
        # 6. تقرير إحصائي
        report_file = self.output_dir / f"{stem}_statistics.md"
        self._export_statistics_report(report_file)
        output_files['statistics'] = report_file
        logger.info(f"تم تصدير: {report_file.name}")
        
        return output_files
    
    def _export_statistics_report(self, output_file: Path):
        """
        تصدير تقرير إحصائي شامل
        
        Args:
            output_file: مسار ملف التقرير
        """
        time_dist = Counter(s.time_of_day for s in self.scenes)
        loc_dist = Counter(s.location_type for s in self.scenes)
        top_chars = sorted(self.character_stats.items(), key=lambda x: x[1], reverse=True)[:15]
        
        avg_dialogues = self.analysis.total_dialogues / self.analysis.total_scenes if self.analysis.total_scenes > 0 else 0
        
        report = f"""# تقرير تحليل السيناريو (Docling Full Pipeline)

## المعلومات الأساسية

| الحقل | القيمة |
|-------|--------|
| **العنوان** | {self.analysis.title} |
| **المسلسل** | {self.analysis.series_name} |
| **المؤلف** | {self.analysis.author} |
| **المخرج** | {self.analysis.director} |
| **السنة** | {self.analysis.year} |

## معلومات المعالجة (Docling)

| الحقل | القيمة |
|-------|--------|
| **وقت المعالجة** | {self.analysis.processing_time:.2f} ثانية |
| **عدد الصفحات** | {self.analysis.total_pages} |
| **محرك OCR** | EasyOCR (ar/en) |
| **نموذج الجداول** | TableFormer (ACCURATE) |

## الإحصائيات العامة

| المقياس | القيمة |
|---------|--------|
| **عدد المشاهد** | {self.analysis.total_scenes} |
| **عدد الحوارات** | {self.analysis.total_dialogues} |
| **عدد الشخصيات** | {self.analysis.total_characters} |
| **عدد الجداول** | {self.analysis.total_tables} |
| **عدد الصور** | {self.analysis.total_pictures} |
| **متوسط الحوارات/مشهد** | {avg_dialogues:.2f} |

## عناصر الوثيقة (Layout Analysis)

| نوع العنصر | العدد |
|------------|-------|
"""
        for item_type, items in self.body_items_by_type.items():
            if len(items) > 0:
                report += f"| {item_type} | {len(items)} |\n"
        
        report += """
## توزيع المشاهد

### حسب الوقت
"""
        for time, count in time_dist.items():
            pct = (count / self.analysis.total_scenes * 100) if self.analysis.total_scenes > 0 else 0
            report += f"- **{time}**: {count} مشهد ({pct:.1f}%)\n"
        
        report += "\n### حسب المكان\n"
        for loc, count in loc_dist.items():
            pct = (count / self.analysis.total_scenes * 100) if self.analysis.total_scenes > 0 else 0
            report += f"- **{loc}**: {count} مشهد ({pct:.1f}%)\n"
        
        report += "\n## أكثر 15 شخصية ظهوراً\n\n"
        report += "| الترتيب | الشخصية | عدد الحوارات |\n"
        report += "|---------|----------|-------------|\n"
        for rank, (char, count) in enumerate(top_chars, 1):
            report += f"| {rank} | {char} | {count} |\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def run_full_pipeline(self) -> Dict[str, Path]:
        """
        تشغيل خط الأنابيب الكامل
        
        Returns:
            قاموس بمسارات الملفات المُصدَّرة
        """
        logger.info("=" * 70)
        logger.info("Docling Full Pipeline - معالجة السيناريو العربي")
        logger.info("=" * 70)
        
        # 1. معالجة الوثيقة
        self.process_document()
        
        # 2. تحليل البنية
        self.analyze_document_structure()
        
        # 3. استخراج المعلومات الوصفية
        self.extract_metadata()
        
        # 4. استخراج المشاهد والحوارات
        self.extract_scenes_and_dialogues()
        
        # 5. تصدير جميع التنسيقات
        output_files = self.export_all_formats()
        
        logger.info("=" * 70)
        logger.info("اكتملت المعالجة بنجاح!")
        logger.info("=" * 70)
        
        return output_files


def main():
    """الدالة الرئيسية"""

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="مسار ملف PDF المدخل")
    ap.add_argument("--output_dir", default="docling_output", help="مجلد المخرجات")
    args = ap.parse_args()

    input_file = str(args.input)
    if not Path(input_file).exists():
        logger.error(f"الملف غير موجود: {input_file}")
        return

    pipeline = DoclingFullPipeline(
        input_file=input_file,
        output_dir=str(args.output_dir),
    )
    
    try:
        # تشغيل خط الأنابيب الكامل
        output_files = pipeline.run_full_pipeline()
        
        print("\n" + "=" * 70)
        print("✓ تم إنشاء Dataset بنجاح باستخدام Docling Full Pipeline!")
        print("=" * 70)
        print("\nالملفات المُنشأة:")
        for name, path in output_files.items():
            print(f"  • {name}: {path}")
        print("\n" + "=" * 70)
        
    except Exception as e:
        logger.error(f"فشل تشغيل خط الأنابيب: {e}")
        raise


if __name__ == "__main__":
    main()
