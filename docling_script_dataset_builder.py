"""
سكريبت إعداد Dataset من ملفات السيناريو باستخدام Docling
يقوم بتحويل TXT إلى Markdown ثم معالجته باستخدام Docling
"""

import re
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

# إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('docling_dataset_builder.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """فئة تمثل مشهد واحد في السيناريو"""
    scene_number: int
    time_of_day: str
    location_type: str
    location_name: str
    content: str
    dialogues: List[Dict[str, str]]
    characters: List[str]
    stage_directions: List[str]
    line_start: int
    line_end: int


@dataclass
class ScriptMetadata:
    """معلومات وصفية عن السيناريو"""
    title: str
    author: str
    director: str
    series_name: str
    episode_number: str
    year: str
    total_scenes: int
    total_characters: int
    total_dialogues: int
    total_words: int


class DoclingScriptDatasetBuilder:
    """فئة بناء Dataset من ملفات السيناريو باستخدام Docling"""
    
    def __init__(self, input_file: str, output_dir: str = "dataset_output"):
        """
        تهيئة بناء Dataset
        
        Args:
            input_file: مسار ملف السيناريو المدخل (TXT)
            output_dir: مجلد المخرجات
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.raw_text = ""
        self.markdown_file: Optional[Path] = None
        self.docling_result = None
        self.scenes: List[Scene] = []
        self.metadata: ScriptMetadata = None
        self.character_stats: Dict[str, int] = {}
        
        logger.info(f"تم تهيئة بناء Dataset للملف: {self.input_file.name}")
    
    def read_script_file(self) -> str:
        """
        قراءة ملف السيناريو مع معالجة الترميز
        
        Returns:
            النص الكامل للسيناريو
        """
        try:
            encodings = ['utf-8', 'utf-8-sig', 'windows-1256', 'cp1256']
            
            for encoding in encodings:
                try:
                    with open(self.input_file, 'r', encoding=encoding) as f:
                        self.raw_text = f.read()
                    logger.info(f"تم قراءة الملف بنجاح باستخدام ترميز: {encoding}")
                    return self.raw_text
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("فشل قراءة الملف بجميع الترميزات المتاحة")
            
        except Exception as e:
            logger.error(f"خطأ في قراءة الملف: {e}")
            raise
    
    def convert_to_structured_markdown(self) -> Path:
        """
        تحويل ملف TXT إلى Markdown منظم
        
        Returns:
            مسار ملف Markdown المُنشأ
        """
        lines = self.raw_text.split('\n')
        markdown_lines = []
        
        # استخراج المعلومات الأساسية من الرأس
        title = ""
        author = ""
        director = ""
        series_name = ""
        year = ""
        
        # معالجة الرأس
        header_processed = False
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            
            if i == 0 and line:
                title = line
                markdown_lines.append(f"# {title}\n")
            elif 'تأليف' in line:
                author = re.sub(r'تأليف|المؤلف', '', line).strip()
                markdown_lines.append(f"**تأليف**: {author}\n")
            elif 'إخراج' in line:
                director = re.sub(r'إخراج|المخرج', '', line).strip()
                markdown_lines.append(f"**إخراج**: {director}\n")
            elif 'الشعلة' in line:
                series_name = line
                markdown_lines.append(f"**المسلسل**: {series_name}\n")
            elif re.search(r'\d{4}', line):
                year_match = re.search(r'\d{4}', line)
                if year_match:
                    year = year_match.group()
                    markdown_lines.append(f"**السنة**: {year}\n")
            elif 'بسم الله' in line:
                markdown_lines.append(f"\n{line}\n")
                markdown_lines.append("\n---\n\n")
                header_processed = True
                break
        
        # معالجة المشاهد
        scene_pattern = re.compile(r'مشهد\s+(\d+)\s+(.*?)\s+(ليل|نهار)-(داخلي|خارجي)')
        in_scene = False
        
        for line in lines[20:]:
            line = line.strip()
            
            if not line:
                markdown_lines.append("\n")
                continue
            
            # التحقق من بداية مشهد جديد
            match = scene_pattern.search(line)
            if match:
                scene_number = match.group(1)
                location_info = match.group(2).strip()
                time_of_day = match.group(3)
                location_type = match.group(4)
                
                markdown_lines.append(f"\n## مشهد {scene_number}\n\n")
                markdown_lines.append(f"**الوقت**: {time_of_day} | **المكان**: {location_type} - {location_info}\n\n")
                in_scene = True
                continue
            
            # التحقق من نهاية المشهد
            if line == 'قطع':
                markdown_lines.append("\n---\n")
                in_scene = False
                continue
            
            # معالجة الحوارات والإرشادات
            dialogue_pattern = re.compile(r'^([^:]+)\s*:\s*(.*)$')
            match = dialogue_pattern.match(line)
            
            if match and in_scene:
                character = match.group(1).strip()
                dialogue = match.group(2).strip()
                
                # التحقق من أن هذا حوار وليس إرشاد
                stage_keywords = ['يجلس', 'يقف', 'يدخل', 'يخرج', 'ينظر', 'يتحدث',
                                'تجلس', 'تقف', 'تدخل', 'تخرج', 'تنظر', 'تتحدث']
                
                is_stage = any(keyword in line for keyword in stage_keywords)
                
                if not is_stage and len(character.split()) <= 3:
                    # حوار
                    if dialogue:
                        markdown_lines.append(f"**{character}**: {dialogue}\n\n")
                    else:
                        markdown_lines.append(f"**{character}**:\n\n")
                else:
                    # إرشاد مسرحي
                    markdown_lines.append(f"*{line}*\n\n")
            else:
                # نص عادي أو إرشاد
                if in_scene:
                    markdown_lines.append(f"{line}\n\n")
                else:
                    markdown_lines.append(f"{line}\n")
        
        # حفظ ملف Markdown
        markdown_file = self.output_dir / f"{self.input_file.stem}.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.writelines(markdown_lines)
        
        self.markdown_file = markdown_file
        logger.info(f"تم تحويل الملف إلى Markdown: {markdown_file}")
        return markdown_file
    
    def process_with_docling(self):
        """
        معالجة ملف Markdown باستخدام Docling
        
        Returns:
            نتيجة معالجة Docling
        """
        try:
            logger.info("بدء معالجة Docling...")
            
            # تهيئة محول Docling
            converter = DocumentConverter()
            
            # معالجة ملف Markdown
            result = converter.convert(str(self.markdown_file))
            
            self.docling_result = result
            logger.info("اكتملت معالجة Docling بنجاح")
            
            return result
            
        except Exception as e:
            logger.error(f"خطأ في معالجة Docling: {e}")
            raise
    
    def extract_metadata_from_docling(self) -> ScriptMetadata:
        """
        استخراج المعلومات الوصفية من نتيجة Docling
        
        Returns:
            كائن ScriptMetadata
        """
        doc = self.docling_result.document
        
        # استخراج المعلومات من النص
        text = doc.export_to_markdown()
        lines = text.split('\n')[:20]
        
        title = ""
        author = ""
        director = ""
        series_name = ""
        year = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                title = line[2:].strip()
            elif 'تأليف' in line:
                author = re.sub(r'\*\*تأليف\*\*:\s*', '', line).strip()
            elif 'إخراج' in line:
                director = re.sub(r'\*\*إخراج\*\*:\s*', '', line).strip()
            elif 'المسلسل' in line:
                series_name = re.sub(r'\*\*المسلسل\*\*:\s*', '', line).strip()
            elif 'السنة' in line:
                year = re.sub(r'\*\*السنة\*\*:\s*', '', line).strip()
        
        self.metadata = ScriptMetadata(
            title=title,
            author=author,
            director=director,
            series_name=series_name,
            episode_number="",
            year=year,
            total_scenes=0,
            total_characters=0,
            total_dialogues=0,
            total_words=0
        )
        
        logger.info(f"تم استخراج المعلومات الوصفية: {title}")
        return self.metadata
    
    def parse_scenes_from_docling(self) -> List[Scene]:
        """
        تحليل واستخراج المشاهد من نتيجة Docling
        
        Returns:
            قائمة بكائنات Scene
        """
        doc = self.docling_result.document
        markdown_text = doc.export_to_markdown()
        lines = markdown_text.split('\n')
        
        scene_pattern = re.compile(r'^##\s+مشهد\s+(\d+)')
        metadata_pattern = re.compile(r'\*\*الوقت\*\*:\s*(\S+)\s*\|\s*\*\*المكان\*\*:\s*(\S+)\s*-\s*(.+)')
        
        current_scene = None
        scene_content = []
        scene_start_line = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # التحقق من بداية مشهد جديد
            scene_match = scene_pattern.match(line)
            
            if scene_match:
                # حفظ المشهد السابق
                if current_scene is not None:
                    current_scene.content = '\n'.join(scene_content)
                    current_scene.line_end = line_num - 1
                    self._analyze_scene_content_docling(current_scene)
                    self.scenes.append(current_scene)
                
                # بدء مشهد جديد
                scene_number = int(scene_match.group(1))
                
                current_scene = Scene(
                    scene_number=scene_number,
                    time_of_day="",
                    location_type="",
                    location_name="",
                    content="",
                    dialogues=[],
                    characters=[],
                    stage_directions=[],
                    line_start=line_num,
                    line_end=0
                )
                
                scene_content = []
                continue
            
            # استخراج معلومات المشهد
            if current_scene and not current_scene.time_of_day:
                meta_match = metadata_pattern.match(line)
                if meta_match:
                    current_scene.time_of_day = meta_match.group(1)
                    current_scene.location_type = meta_match.group(2)
                    current_scene.location_name = meta_match.group(3).strip()
                    continue
            
            # جمع محتوى المشهد
            if current_scene and line and line != '---':
                scene_content.append(line)
        
        # حفظ المشهد الأخير
        if current_scene is not None:
            current_scene.content = '\n'.join(scene_content)
            current_scene.line_end = len(lines)
            self._analyze_scene_content_docling(current_scene)
            self.scenes.append(current_scene)
        
        logger.info(f"تم استخراج {len(self.scenes)} مشهد من Docling")
        return self.scenes
    
    def _analyze_scene_content_docling(self, scene: Scene):
        """
        تحليل محتوى المشهد من Markdown المُنتج بواسطة Docling
        
        Args:
            scene: كائن المشهد المراد تحليله
        """
        lines = scene.content.split('\n')
        
        dialogue_pattern = re.compile(r'^\*\*([^*]+)\*\*:\s*(.*)$')
        stage_direction_pattern = re.compile(r'^\*(.+)\*$')
        
        current_character = None
        current_dialogue = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # التحقق من حوار
            dialogue_match = dialogue_pattern.match(line)
            if dialogue_match:
                # حفظ الحوار السابق
                if current_character and current_dialogue:
                    self._save_dialogue(scene, current_character, current_dialogue)
                
                # بدء حوار جديد
                current_character = dialogue_match.group(1).strip()
                dialogue_text = dialogue_match.group(2).strip()
                current_dialogue = [dialogue_text] if dialogue_text else []
                continue
            
            # التحقق من إرشاد مسرحي
            stage_match = stage_direction_pattern.match(line)
            if stage_match:
                # حفظ الحوار السابق
                if current_character and current_dialogue:
                    self._save_dialogue(scene, current_character, current_dialogue)
                    current_character = None
                    current_dialogue = []
                
                scene.stage_directions.append(stage_match.group(1))
                continue
            
            # استمرار الحوار
            if current_character:
                current_dialogue.append(line)
            else:
                # نص عادي
                if line and not line.startswith('**'):
                    scene.stage_directions.append(line)
        
        # حفظ الحوار الأخير
        if current_character and current_dialogue:
            self._save_dialogue(scene, current_character, current_dialogue)
    
    def _save_dialogue(self, scene: Scene, character: str, dialogue_lines: List[str]):
        """
        حفظ حوار شخصية في المشهد
        
        Args:
            scene: كائن المشهد
            character: اسم الشخصية
            dialogue_lines: أسطر الحوار
        """
        if not dialogue_lines:
            return
        
        dialogue_text = ' '.join(dialogue_lines)
        scene.dialogues.append({
            'character': character,
            'text': dialogue_text
        })
        
        if character not in scene.characters:
            scene.characters.append(character)
        
        if character not in self.character_stats:
            self.character_stats[character] = 0
        self.character_stats[character] += 1
    
    def calculate_statistics(self):
        """حساب الإحصائيات الشاملة للسيناريو"""
        total_dialogues = sum(len(scene.dialogues) for scene in self.scenes)
        total_words = sum(
            len(dialogue['text'].split())
            for scene in self.scenes
            for dialogue in scene.dialogues
        )
        
        self.metadata.total_scenes = len(self.scenes)
        self.metadata.total_characters = len(self.character_stats)
        self.metadata.total_dialogues = total_dialogues
        self.metadata.total_words = total_words
        
        logger.info(f"الإحصائيات: {self.metadata.total_scenes} مشهد، "
                   f"{self.metadata.total_characters} شخصية، "
                   f"{self.metadata.total_dialogues} حوار")
    
    def export_to_json(self):
        """تصدير Dataset بتنسيق JSON"""
        output_file = self.output_dir / f"{self.input_file.stem}_docling_dataset.json"
        
        dataset = {
            'metadata': asdict(self.metadata),
            'docling_info': {
                'pages': len(self.docling_result.document.pages),
                'processing_method': 'Docling + Markdown'
            },
            'scenes': [
                {
                    'scene_number': scene.scene_number,
                    'time_of_day': scene.time_of_day,
                    'location_type': scene.location_type,
                    'location_name': scene.location_name,
                    'dialogues': scene.dialogues,
                    'characters': scene.characters,
                    'stage_directions': scene.stage_directions,
                    'line_range': f"{scene.line_start}-{scene.line_end}"
                }
                for scene in self.scenes
            ],
            'character_statistics': self.character_stats
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"تم تصدير JSON إلى: {output_file}")
        return output_file
    
    def export_to_csv(self):
        """تصدير الحوارات بتنسيق CSV"""
        output_file = self.output_dir / f"{self.input_file.stem}_docling_dialogues.csv"
        
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'رقم_المشهد', 'الوقت', 'المكان_نوع', 'المكان_اسم',
                'الشخصية', 'الحوار', 'عدد_الكلمات'
            ])
            
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
        
        logger.info(f"تم تصدير CSV إلى: {output_file}")
        return output_file
    
    def export_statistics_report(self):
        """تصدير تقرير إحصائي شامل"""
        output_file = self.output_dir / f"{self.input_file.stem}_docling_statistics.md"
        
        time_distribution = Counter(scene.time_of_day for scene in self.scenes)
        location_distribution = Counter(scene.location_type for scene in self.scenes)
        
        top_characters = sorted(
            self.character_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        avg_dialogues_per_scene = (
            self.metadata.total_dialogues / self.metadata.total_scenes
            if self.metadata.total_scenes > 0 else 0
        )
        
        report = f"""# تقرير إحصائي (Docling) - {self.metadata.title}

## المعلومات الأساسية

- **العنوان**: {self.metadata.title}
- **المؤلف**: {self.metadata.author}
- **المخرج**: {self.metadata.director}
- **المسلسل**: {self.metadata.series_name}
- **السنة**: {self.metadata.year}

## معلومات المعالجة

- **طريقة المعالجة**: Docling + Markdown
- **ملف Markdown**: {self.markdown_file.name}

## الإحصائيات العامة

- **عدد المشاهد**: {self.metadata.total_scenes}
- **عدد الشخصيات**: {self.metadata.total_characters}
- **عدد الحوارات**: {self.metadata.total_dialogues}
- **عدد الكلمات**: {self.metadata.total_words:,}
- **متوسط الحوارات لكل مشهد**: {avg_dialogues_per_scene:.2f}

## توزيع المشاهد حسب الوقت

"""
        for time, count in time_distribution.items():
            percentage = (count / self.metadata.total_scenes) * 100
            report += f"- **{time}**: {count} مشهد ({percentage:.1f}%)\n"
        
        report += "\n## توزيع المشاهد حسب المكان\n\n"
        for location, count in location_distribution.items():
            percentage = (count / self.metadata.total_scenes) * 100
            report += f"- **{location}**: {count} مشهد ({percentage:.1f}%)\n"
        
        report += "\n## أكثر 10 شخصيات ظهوراً\n\n"
        report += "| الترتيب | الشخصية | عدد الحوارات |\n"
        report += "|---------|----------|-------------|\n"
        
        for rank, (character, count) in enumerate(top_characters, 1):
            report += f"| {rank} | {character} | {count} |\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"تم تصدير التقرير الإحصائي إلى: {output_file}")
        return output_file
    
    def build_complete_dataset(self):
        """
        بناء Dataset كامل باستخدام Docling
        
        Returns:
            قاموس بمسارات الملفات المُصدَّرة
        """
        logger.info("=" * 60)
        logger.info("بدء عملية بناء Dataset باستخدام Docling")
        logger.info("=" * 60)
        
        # 1. قراءة الملف
        self.read_script_file()
        
        # 2. تحويل إلى Markdown
        self.convert_to_structured_markdown()
        
        # 3. معالجة باستخدام Docling
        self.process_with_docling()
        
        # 4. استخراج المعلومات الوصفية
        self.extract_metadata_from_docling()
        
        # 5. تحليل المشاهد
        self.parse_scenes_from_docling()
        
        # 6. حساب الإحصائيات
        self.calculate_statistics()
        
        # 7. التصدير بتنسيقات متعددة
        json_file = self.export_to_json()
        csv_file = self.export_to_csv()
        stats_file = self.export_statistics_report()
        
        logger.info("=" * 60)
        logger.info("اكتملت عملية بناء Dataset بنجاح!")
        logger.info("=" * 60)
        
        return {
            'markdown': self.markdown_file,
            'json': json_file,
            'csv': csv_file,
            'statistics': stats_file
        }


def main():
    """الدالة الرئيسية لتشغيل السكريبت"""
    
    input_file = r"E:\PREPA\Extracted_Dataset\7.txt"
    
    builder = DoclingScriptDatasetBuilder(
        input_file=input_file,
        output_dir="E:/PREPA/Extracted_Dataset/docling_dataset_output"
    )
    
    try:
        output_files = builder.build_complete_dataset()
        
        print("\n" + "=" * 60)
        print("✓ تم إنشاء Dataset بنجاح باستخدام Docling!")
        print("=" * 60)
        print("\nالملفات المُنشأة:")
        print(f"  • Markdown: {output_files['markdown']}")
        print(f"  • JSON Dataset: {output_files['json']}")
        print(f"  • CSV Dialogues: {output_files['csv']}")
        print(f"  • Statistics Report: {output_files['statistics']}")
        print("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"فشلت عملية بناء Dataset: {e}")
        raise


if __name__ == "__main__":
    main()
