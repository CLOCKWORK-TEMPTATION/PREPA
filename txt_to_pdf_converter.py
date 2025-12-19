#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    محول الملفات النصية إلى PDF                              ║
║                    Text to PDF High-Quality Converter                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

الوصف: يقوم هذا السكريبت بتحويل ملفات TXT إلى ملفات PDF عالية الجودة
       مع دعم كامل للغة العربية والإنجليزية

المتطلبات:
    pip install reportlab arabic-reshaper python-bidi

الاستخدام:
    python txt_to_pdf_converter.py
"""

import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# إعداد نظام التسجيل (Logging)
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('conversion.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# التحقق من المكتبات المطلوبة وتثبيتها
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dependencies() -> bool:
    """التحقق من وجود المكتبات المطلوبة وتثبيتها إذا لزم الأمر"""
    required_packages = {
        'reportlab': 'reportlab',
        'arabic_reshaper': 'arabic-reshaper',
        'bidi': 'python-bidi'
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.warning(f"المكتبات التالية غير موجودة: {', '.join(missing_packages)}")
        logger.info("جاري تثبيت المكتبات المطلوبة...")
        
        import subprocess
        for package in missing_packages:
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info(f"تم تثبيت: {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"فشل تثبيت {package}: {e}")
                return False
    
    return True


# ══════════════════════════════════════════════════════════════════════════════
# فئة إعدادات التحويل
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversionConfig:
    """إعدادات تحويل PDF"""
    
    # أبعاد الصفحة (A4 بالنقاط)
    page_width: float = 595.27
    page_height: float = 841.89
    
    # الهوامش
    margin_top: float = 72.0      # 1 بوصة
    margin_bottom: float = 72.0
    margin_left: float = 72.0
    margin_right: float = 72.0
    
    # إعدادات الخط
    font_size: float = 12.0
    line_height: float = 18.0
    
    # إعدادات الجودة
    title: str = "Converted Document"
    author: str = "TXT to PDF Converter"
    
    # مجلد الإخراج
    output_dir: Optional[Path] = None
    
    # قائمة الملفات المصدر
    source_files: list = field(default_factory=list)
    
    @property
    def content_width(self) -> float:
        """عرض منطقة المحتوى"""
        return self.page_width - self.margin_left - self.margin_right
    
    @property
    def content_height(self) -> float:
        """ارتفاع منطقة المحتوى"""
        return self.page_height - self.margin_top - self.margin_bottom


# ══════════════════════════════════════════════════════════════════════════════
# فئة معالج النصوص العربية
# ══════════════════════════════════════════════════════════════════════════════

class ArabicTextProcessor:
    """معالج النصوص العربية للتحويل الصحيح في PDF"""
    
    def __init__(self):
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            self._reshaper = arabic_reshaper
            self._get_display = get_display
            self._arabic_support = True
            logger.info("تم تفعيل دعم اللغة العربية")
        except ImportError:
            self._arabic_support = False
            logger.warning("دعم العربية غير متاح - سيتم عرض النص كما هو")
    
    def process(self, text: str) -> str:
        """معالجة النص للعرض الصحيح في PDF"""
        if not self._arabic_support:
            return text
        
        # التحقق من وجود أحرف عربية
        if any('\u0600' <= char <= '\u06FF' for char in text):
            try:
                reshaped = self._reshaper.reshape(text)
                return self._get_display(reshaped)
            except Exception as e:
                logger.debug(f"خطأ في معالجة النص العربي: {e}")
                return text
        
        return text


# ══════════════════════════════════════════════════════════════════════════════
# فئة محول PDF
# ══════════════════════════════════════════════════════════════════════════════

class PDFConverter:
    """المحول الرئيسي من TXT إلى PDF"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.arabic_processor = ArabicTextProcessor()
        
        # استيراد مكتبات ReportLab
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.units import inch
        
        self._canvas_module = canvas
        self._pdfmetrics = pdfmetrics
        self._TTFont = TTFont
        
        # تسجيل الخطوط
        self._register_fonts()
    
    def _register_fonts(self) -> None:
        """تسجيل الخطوط المتاحة"""
        # محاولة استخدام خط يدعم العربية
        font_paths = [
            # Windows
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/times.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            # macOS
            "/Library/Fonts/Arial.ttf",
        ]
        
        self.font_name = "Helvetica"  # الخط الافتراضي
        
        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    self._pdfmetrics.registerFont(
                        self._TTFont('CustomFont', font_path)
                    )
                    self.font_name = 'CustomFont'
                    logger.info(f"تم تسجيل الخط: {font_path}")
                    break
                except Exception as e:
                    logger.debug(f"فشل تسجيل الخط {font_path}: {e}")
    
    def _read_text_file(self, file_path: Path) -> Optional[str]:
        """قراءة محتوى الملف النصي"""
        encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-6', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.debug(f"تم قراءة الملف بترميز: {encoding}")
                return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.error(f"خطأ في قراءة الملف {file_path}: {e}")
                return None
        
        logger.error(f"فشل قراءة الملف {file_path} بجميع الترميزات المتاحة")
        return None
    
    def _wrap_text(self, text: str, max_width: float, font_name: str, font_size: float) -> list:
        """تقسيم النص إلى أسطر تناسب عرض الصفحة"""
        from reportlab.pdfbase.pdfmetrics import stringWidth
        
        lines = []
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                lines.append('')
                continue
            
            words = paragraph.split()
            if not words:
                lines.append('')
                continue
            
            current_line = []
            current_width = 0
            
            for word in words:
                word_width = stringWidth(word, font_name, font_size)
                space_width = stringWidth(' ', font_name, font_size)
                
                if current_width + word_width <= max_width:
                    current_line.append(word)
                    current_width += word_width + space_width
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_width + space_width
            
            if current_line:
                lines.append(' '.join(current_line))
        
        return lines
    
    def convert_file(self, input_path: Path, output_path: Path) -> bool:
        """تحويل ملف نصي واحد إلى PDF"""
        try:
            # قراءة المحتوى
            content = self._read_text_file(input_path)
            if content is None:
                return False
            
            # إنشاء PDF
            c = self._canvas_module.Canvas(
                str(output_path),
                pagesize=(self.config.page_width, self.config.page_height)
            )
            
            # إعداد البيانات الوصفية
            c.setTitle(input_path.stem)
            c.setAuthor(self.config.author)
            c.setCreator("TXT to PDF Converter")
            c.setSubject(f"Converted from {input_path.name}")
            
            # إعداد الخط
            c.setFont(self.font_name, self.config.font_size)
            
            # معالجة النص
            processed_content = self.arabic_processor.process(content)
            
            # تقسيم النص إلى أسطر
            lines = self._wrap_text(
                processed_content,
                self.config.content_width,
                self.font_name,
                self.config.font_size
            )
            
            # كتابة النص
            y_position = self.config.page_height - self.config.margin_top
            
            for line in lines:
                # التحقق من الحاجة لصفحة جديدة
                if y_position < self.config.margin_bottom:
                    c.showPage()
                    c.setFont(self.font_name, self.config.font_size)
                    y_position = self.config.page_height - self.config.margin_top
                
                # كتابة السطر
                c.drawString(self.config.margin_left, y_position, line)
                y_position -= self.config.line_height
            
            # حفظ PDF
            c.save()
            
            logger.info(f"تم التحويل: {input_path.name} -> {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"فشل تحويل {input_path}: {e}")
            return False
    
    def convert_all(self) -> dict:
        """تحويل جميع الملفات المحددة"""
        results = {
            'success': [],
            'failed': [],
            'total': len(self.config.source_files)
        }
        
        # تحديد مجلد الإخراج
        if self.config.output_dir is None:
            if self.config.source_files:
                self.config.output_dir = Path(self.config.source_files[0]).parent
            else:
                self.config.output_dir = Path.cwd()
        
        # إنشاء مجلد الإخراج إذا لم يكن موجوداً
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"بدء تحويل {results['total']} ملف(ات)")
        logger.info(f"مجلد الإخراج: {self.config.output_dir}")
        
        for file_path in self.config.source_files:
            input_path = Path(file_path)
            
            if not input_path.exists():
                logger.warning(f"الملف غير موجود: {input_path}")
                results['failed'].append(str(input_path))
                continue
            
            output_path = self.config.output_dir / f"{input_path.stem}.pdf"
            
            if self.convert_file(input_path, output_path):
                results['success'].append(str(output_path))
            else:
                results['failed'].append(str(input_path))
        
        return results


# ══════════════════════════════════════════════════════════════════════════════
# الدالة الرئيسية
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    """نقطة الدخول الرئيسية"""
    
    logger.info("=" * 60)
    logger.info("محول الملفات النصية إلى PDF")
    logger.info("=" * 60)
    
    # التحقق من المكتبات
    if not ensure_dependencies():
        logger.error("فشل تثبيت المكتبات المطلوبة")
        return 1
    
    # تحديد الملفات المصدر
    source_files = [
        r"E:\PREPA\Extracted_Dataset\1.txt",
        r"E:\PREPA\Extracted_Dataset\2.txt",
        r"E:\PREPA\Extracted_Dataset\3.txt",
        r"E:\PREPA\Extracted_Dataset\4.txt",
        r"E:\PREPA\Extracted_Dataset\5.txt",
        r"E:\PREPA\Extracted_Dataset\6.txt",
        r"E:\PREPA\Extracted_Dataset\7.txt",
        r"E:\PREPA\Extracted_Dataset\8.txt",
        r"E:\PREPA\Extracted_Dataset\9.txt",
        r"E:\PREPA\Extracted_Dataset\10.txt",
    ]
    
    # إعداد التكوين
    config = ConversionConfig(
        source_files=source_files,
        output_dir=Path(r"E:\PREPA\Extracted_Dataset"),  # نفس المجلد
        font_size=12.0,
        line_height=18.0,
        margin_top=72.0,
        margin_bottom=72.0,
        margin_left=72.0,
        margin_right=72.0,
    )
    
    # تنفيذ التحويل
    try:
        converter = PDFConverter(config)
        results = converter.convert_all()
        
        # عرض النتائج
        logger.info("=" * 60)
        logger.info("نتائج التحويل:")
        logger.info(f"  الإجمالي: {results['total']}")
        logger.info(f"  النجاح:   {len(results['success'])}")
        logger.info(f"  الفشل:    {len(results['failed'])}")
        
        if results['success']:
            logger.info("الملفات المحولة بنجاح:")
            for path in results['success']:
                logger.info(f"  ✓ {path}")
        
        if results['failed']:
            logger.warning("الملفات التي فشل تحويلها:")
            for path in results['failed']:
                logger.warning(f"  ✗ {path}")
        
        logger.info("=" * 60)
        
        return 0 if not results['failed'] else 1
        
    except Exception as e:
        logger.exception(f"خطأ غير متوقع: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
