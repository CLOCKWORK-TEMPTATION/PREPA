=======================================
(venv) PS E:\PREPA> python docling_script_dataset_builder.py
Traceback (most recent call last):
  File "E:\PREPA\docling_script_dataset_builder.py", line 15, in <module>
    from docling.document_converter import DocumentConverter
ModuleNotFoundError: No module named 'docling'
(venv) PS E:\PREPA> """
سكريبت إعداد Dataset من ملفات السيناريو
يقوم بتحليل ملفات السيناريو واستخراج البيانات المنظمة
"""

import re
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

# إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_builder.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """فئة تمثل مشهد واحد في السيناريو"""
    scene_number: int
    time_of_day: str  # ليل/نهار
    location_type: str  # داخلي/خارجي
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


class ScriptDatasetBuilder:
    """فئة بناء Dataset من ملفات السيناريو"""
    
    def __init__(self, input_file: str, output_dir: str = "dataset_output"):
        """
        تهيئة بناء Dataset
        
        Args:
            input_file: مسار ملف السيناريو المدخل
            output_dir: مجلد المخرجات
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.raw_text = ""
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
            # محاولة القراءة بترميزات مختلفة
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
    
    def extract_metadata(self) -> ScriptMetadata:
        """
        استخراج المعلومات الوصفية من رأس السيناريو
        
        Returns:
            كائن ScriptMetadata
        """
        lines = self.raw_text.split('\n')
        
        # استخراج المعلومات الأساسية من أول 20 سطر
        title = ""
        author = ""
        director = ""
        series_name = ""
        episode_number = ""
        year = ""
        
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            
            if i == 0:
                title = line
            elif 'تأليف' in line or 'المؤلف' in line:
                author = re.sub(r'تأليف|المؤلف', '', line).strip()
            elif 'إخراج' in line or 'المخرج' in line:
                director = re.sub(r'إخراج|المخرج', '', line).strip()
            elif 'الشعلة' in line:
                series_name = line
            elif re.search(r'\d{4}', line):
                year_match = re.search(r'\d{4}', line)
                if year_match:
                    year = year_match.group()
        
        # سيتم تحديث هذه القيم بعد معالجة المشاهد
        self.metadata = ScriptMetadata(
            title=title,
            author=author,
            director=director,
            series_name=series_name,
            episode_number=episode_number,
            year=year,
            total_scenes=0,
            total_characters=0,
            total_dialogues=0,
            total_words=0
        )
        
        logger.info(f"تم استخراج المعلومات الوصفية: {title}")
        return self.metadata
    
    def parse_scenes(self) -> List[Scene]:
        """
        تحليل واستخراج المشاهد من السيناريو
        
        Returns:
            قائمة بكائنات Scene
        """
        lines = self.raw_text.split('\n')
        
        # نمط للتعرف على بداية المشهد
        scene_pattern = re.compile(r'مشهد\s+(\d+)\s+(.*?)\s+(ليل|نهار)-(داخلي|خارجي)')
        
        current_scene = None
        scene_content = []
        scene_start_line = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # التحقق من بداية مشهد جديد
            match = scene_pattern.search(line)
            
            if match:
                # حفظ المشهد السابق إذا كان موجوداً
                if current_scene is not None:
                    current_scene.content = '\n'.join(scene_content)
                    current_scene.line_end = line_num - 1
                    self._analyze_scene_content(current_scene)
                    self.scenes.append(current_scene)
                
                # بدء مشهد جديد
                scene_number = int(match.group(1))
                location_info = match.group(2).strip()
                time_of_day = match.group(3)
                location_type = match.group(4)
                
                current_scene = Scene(
                    scene_number=scene_number,
                    time_of_day=time_of_day,
                    location_type=location_type,
                    location_name=location_info,
                    content="",
                    dialogues=[],
                    characters=[],
                    stage_directions=[],
                    line_start=line_num,
                    line_end=0
                )
                
                scene_content = []
                scene_start_line = line_num
                
            elif current_scene is not None and line and line != 'قطع':
                scene_content.append(line)
        
        # حفظ المشهد الأخير
        if current_scene is not None:
            current_scene.content = '\n'.join(scene_content)
            current_scene.line_end = len(lines)
            self._analyze_scene_content(current_scene)
            self.scenes.append(current_scene)
        
        logger.info(f"تم استخراج {len(self.scenes)} مشهد")
        return self.scenes
    
    def _analyze_scene_content(self, scene: Scene):
        """
        تحليل محتوى المشهد لاستخراج الحوارات والشخصيات
        
        Args:
            scene: كائن المشهد المراد تحليله
        """
        lines = scene.content.split('\n')
        
        # أنماط للتعرف على الحوار والإرشادات
        dialogue_with_colon = re.compile(r'^([^:]+)\s*:\s*(.*)$')  # اسم : حوار
        character_name_only = re.compile(r'^([أ-يa-zA-Z\s]+)\s*:\s*$')  # اسم : فقط
        stage_direction_keywords = ['يجلس', 'يقف', 'يدخل', 'يخرج', 'ينظر', 'يتحدث', 
                                   'تجلس', 'تقف', 'تدخل', 'تخرج', 'تنظر', 'تتحدث',
                                   'ينهض', 'تنهض', 'يمشي', 'تمشي', 'يرقد', 'ترقد',
                                   'بينما', 'فجأة', 'ثم', 'وهو', 'وهي', 'متنهدة', 'بحدة',
                                   'باستغراب', 'مستغربا', 'بفرحة', 'غاضبا']
        
        current_character = None
        current_dialogue = []
        in_dialogue_mode = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # التحقق من اسم شخصية فقط (بدون حوار)
            char_only_match = character_name_only.match(line)
            if char_only_match:
                # حفظ الحوار السابق
                if current_character and current_dialogue:
                    self._save_dialogue(scene, current_character, current_dialogue)
                
                # بدء حوار جديد
                current_character = char_only_match.group(1).strip()
                current_dialogue = []
                in_dialogue_mode = True
                continue
            
            # التحقق من اسم شخصية مع حوار
            dialogue_match = dialogue_with_colon.match(line)
            if dialogue_match:
                potential_character = dialogue_match.group(1).strip()
                dialogue_text = dialogue_match.group(2).strip()
                
                # التحقق من أن هذا ليس إرشاد مسرحي
                is_stage_direction = any(keyword in line for keyword in stage_direction_keywords)
                
                # إذا كان السطر قصير جداً أو يحتوي على كلمات مفتاحية للإرشادات، فهو إرشاد
                if not is_stage_direction and len(potential_character.split()) <= 3:
                    # حفظ الحوار السابق
                    if current_character and current_dialogue:
                        self._save_dialogue(scene, current_character, current_dialogue)
                    
                    # بدء حوار جديد
                    current_character = potential_character
                    current_dialogue = [dialogue_text] if dialogue_text else []
                    in_dialogue_mode = True
                    continue
            
            # التحقق من إرشاد مسرحي
            is_stage_direction = any(keyword in line for keyword in stage_direction_keywords)
            
            if is_stage_direction:
                # حفظ الحوار السابق إذا كان موجوداً
                if current_character and current_dialogue:
                    self._save_dialogue(scene, current_character, current_dialogue)
                    current_character = None
                    current_dialogue = []
                    in_dialogue_mode = False
                
                # حفظ الإرشاد المسرحي
                scene.stage_directions.append(line)
            elif in_dialogue_mode and current_character:
                # استمرار الحوار
                current_dialogue.append(line)
            else:
                # إرشاد مسرحي افتراضي
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
        
        # إضافة الشخصية إلى القائمة
        if character not in scene.characters:
            scene.characters.append(character)
        
        # تحديث إحصائيات الشخصيات
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
        
        # تحديث المعلومات الوصفية
        self.metadata.total_scenes = len(self.scenes)
        self.metadata.total_characters = len(self.character_stats)
        self.metadata.total_dialogues = total_dialogues
        self.metadata.total_words = total_words
        
        logger.info(f"الإحصائيات: {self.metadata.total_scenes} مشهد، "
                   f"{self.metadata.total_characters} شخصية، "
                   f"{self.metadata.total_dialogues} حوار")
    
    def export_to_json(self):
        """تصدير Dataset بتنسيق JSON"""
        output_file = self.output_dir / f"{self.input_file.stem}_dataset.json"
        
        dataset = {
            'metadata': asdict(self.metadata),
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
        output_file = self.output_dir / f"{self.input_file.stem}_dialogues.csv"
        
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
        """تصدير تقرير إحصائي شامل بتنسيق Markdown"""
        output_file = self.output_dir / f"{self.input_file.stem}_statistics.md"
        
        # حساب إحصائيات إضافية
        time_distribution = Counter(scene.time_of_day for scene in self.scenes)
        location_distribution = Counter(scene.location_type for scene in self.scenes)
        
        # أكثر الشخصيات ظهوراً
        top_characters = sorted(
            self.character_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # متوسط الحوارات لكل مشهد
        avg_dialogues_per_scene = (
            self.metadata.total_dialogues / self.metadata.total_scenes
            if self.metadata.total_scenes > 0 else 0
        )
        
        report = f"""# تقرير إحصائي - {self.metadata.title}

## المعلومات الأساسية

- **العنوان**: {self.metadata.title}
- **المؤلف**: {self.metadata.author}
- **المخرج**: {self.metadata.director}
- **المسلسل**: {self.metadata.series_name}
- **السنة**: {self.metadata.year}

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
        بناء Dataset كامل من السيناريو
        
        Returns:
            قاموس بمسارات الملفات المُصدَّرة
        """
        logger.info("=" * 60)
        logger.info("بدء عملية بناء Dataset")
        logger.info("=" * 60)
        
        # 1. قراءة الملف
        self.read_script_file()
        
        # 2. استخراج المعلومات الوصفية
        self.extract_metadata()
        
        # 3. تحليل المشاهد
        self.parse_scenes()
        
        # 4. حساب الإحصائيات
        self.calculate_statistics()
        
        # 5. التصدير بتنسيقات متعددة
        json_file = self.export_to_json()
        csv_file = self.export_to_csv()
        stats_file = self.export_statistics_report()
        
        logger.info("=" * 60)
        logger.info("اكتملت عملية بناء Dataset بنجاح!")
        logger.info("=" * 60)
        
        return {
            'json': json_file,
            'csv': csv_file,
            'statistics': stats_file
        }


def main():
    """الدالة الرئيسية لتشغيل السكريبت"""
    
    # تحديد ملف الإدخال
    input_file = r"E:\PREPA\Extracted_Dataset\7.txt"
    
    # إنشاء بناء Dataset
    builder = ScriptDatasetBuilder(
        input_file=input_file,
        output_dir="E:/PREPA/Extracted_Dataset/dataset_output"
    )
    
    try:
        # بناء Dataset الكامل
        output_files = builder.build_complete_dataset()
        
        print("\n" + "=" * 60)
        print("✓ تم إنشاء Dataset بنجاح!")
        print("=" * 60)
        print("\nالملفات المُنشأة:")
        print(f"  • JSON Dataset: {output_files['json']}")
        print(f"  • CSV Dialogues: {output_files['csv']}")
        print(f"  • Statistics Report: {output_files['statistics']}")
        print("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"فشلت عملية بناء Dataset: {e}")
        raise


if __name__ == "__main__":
    main()
