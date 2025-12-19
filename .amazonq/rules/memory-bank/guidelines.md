# إرشادات التطوير

## معايير جودة الكود

### بنية الملفات والتسمية

#### أسماء الملفات الوصفية (100% من الملفات)
- **النمط المتبع**: استخدام أسماء واضحة موجهة نحو الغرض
- **أمثلة من المشروع**:
  - `docling_comprehensive_test.py` - اختبار شامل لجميع الميزات
  - `docling_arabic_rtl_test.py` - اختبار دعم اللغة العربية
  - `screenplay_to_dataset.py` - تحويل السيناريوهات إلى مجموعات بيانات
  - `txt_to_pdf_converter.py` - تحويل النصوص إلى PDF

#### تعليقات الرأس والوثائق (80% من الملفات)
```python
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    محول الملفات النصية إلى PDF                              ║
║                    Text to PDF High-Quality Converter                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

الوصف: يقوم هذا السكريبت بتحويل ملفات TXT إلى ملفات PDF عالية الجودة
       مع دعم كامل للغة العربية والإنجليزية
"""
```

#### تسمية الدوال الوصفية (100% من الملفات)
- **النمط**: أسماء دوال تشير بوضوح إلى الوظيفة
- **أمثلة**:
  - `comprehensive_layout_test()` - اختبار التخطيط الشامل
  - `test_arabic_rtl()` - اختبار النصوص العربية RTL
  - `run_full_pipeline()` - تشغيل خط الأنابيب الكامل
  - `ensure_dependencies()` - التحقق من التبعيات

### تنظيم الاستيراد

#### ترتيب الاستيرادات المعياري (100% من الملفات)
```python
# 1. المكتبة القياسية أولاً
import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

# 2. المكتبات الخارجية
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
```

#### استيرادات محددة (90% من الملفات)
- **تجنب**: `from module import *`
- **المفضل**: استيرادات محددة لتحسين الوضوح والأداء
```python
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
```

### معايير تنسيق الكود

#### المسافة البادئة المتسقة (100% من الملفات)
- **المعيار**: 4 مسافات في جميع الملفات
- **لا توجد استثناءات**: حتى في الكود المتداخل

#### طول السطر وكسر الأسطر (100% من الملفات)
```python
# كسر العناوين الطويلة بشكل مناسب
source = 'https://arxiv.org/pdf/2408.09869'

# كسر المعاملات الطويلة
pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    ocr_options=EasyOcrOptions(lang=["ar", "en"]),
    do_table_structure=True
)
```

#### المسافات البيضاء المتسقة (100% من الملفات)
```python
# مسافات حول العوامل
result = converter.convert(source)
total_pages = len(result.document.pages)

# مسافات بعد الفواصل
languages = ["ar", "en", "fr"]
```

## الأنماط الدلالية

### نمط معالجة المستندات (100% من ملفات Docling)

#### البنية الأساسية المتكررة
```python
def document_processing_function():
    # 1. تهيئة المحول
    converter = DocumentConverter()
    
    # 2. تحديد المصدر
    source = 'https://arxiv.org/pdf/...'
    
    # 3. معالجة المستند مع معالجة الأخطاء
    try:
        result = converter.convert(source)
        # معالجة النتائج والتصدير
    except Exception as e:
        print(f'خطأ: {e}')
```

#### تكوين خيارات المعالجة (60% من الملفات)
```python
# إعداد خيارات متقدمة
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = EasyOcrOptions(
    lang=["ar", "en"],
    force_full_page_ocr=False
)

converter = DocumentConverter(
    format_options={'pdf': pipeline_options}
)
```

### نمط معالجة الأخطاء (100% من الملفات)

#### Try-Catch الشامل
```python
try:
    result = converter.convert(source)
    # معالجة النتائج
except Exception as e:
    logger.error(f'خطأ في التحويل: {e}')
    # آليات التعافي أو الإبلاغ
```

#### رسائل خطأ وصفية باستخدام f-string (100% من الملفات)
```python
logger.error(f"فشل تحويل {input_path}: {e}")
print(f'Error with {source}: {e}')
logger.warning(f"الملف غير موجود: {input_path}")
```

#### التدهور التدريجي (40% من الملفات)
```python
# محاولة ترميزات متعددة
encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-6']
for encoding in encodings:
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        break
    except UnicodeDecodeError:
        continue
```

### نمط إنشاء المخرجات (90% من الملفات)

#### التصدير متعدد التنسيقات
```python
# استخراج تنسيقات متعددة
markdown_output = result.document.export_to_markdown()
json_output = result.document.export_to_json()

# الحفظ مع ترميز UTF-8
with open('output.md', 'w', encoding='utf-8') as f:
    f.write(markdown_output)

with open('output.json', 'w', encoding='utf-8') as f:
    f.write(json_output)
```

#### تأكيد إنشاء الملفات (100% من الملفات)
```python
print('=== Files Created ===')
print('- comprehensive_layout.md')
print('- comprehensive_layout.json')

logger.info(f"تم تصدير: {json_dataset_file.name}")
```

### نمط تقارير التقدم (100% من الملفات)

#### رؤوس الأقسام بمحددات واضحة
```python
print('=== Starting Comprehensive Layout Test ===')
logger.info("=" * 60)
logger.info("محول الملفات النصية إلى PDF")
logger.info("=" * 60)
```

#### رسائل وصفية غنية بالمعلومات
```python
logger.info("جاري تحليل بنية الوثيقة...")
print(f'Total pages: {len(result.document.pages)}')
logger.info(f"تم التحويل: {input_path.name} -> {output_path.name}")
```

#### تحديثات الحالة المرحلية
```python
logger.info("1. معالجة الوثيقة")
logger.info("2. تحليل البنية") 
logger.info("3. استخراج المعلومات الوصفية")
logger.info("4. استخراج المشاهد والحوارات")
logger.info("5. تصدير جميع التنسيقات")
```

### نمط قياس الأداء (60% من الملفات)

#### قياس وقت المعالجة
```python
import time
start_time = time.time()
# كود المعالجة
end_time = time.time()
processing_time = end_time - start_time
logger.info(f'وقت المعالجة: {processing_time:.2f} ثانية')
```

#### إحصائيات مفصلة
```python
print(f'Total text blocks: {text_count}')
print(f'Total tables: {table_count}')
print(f'Total images: {image_count}')
```

## أنماط استخدام API الداخلي

### تكوين DocumentConverter

#### الاستخدام الأساسي (60% من الملفات)
```python
converter = DocumentConverter()
```

#### مع خيارات خط الأنابيب (40% من الملفات)
```python
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = EasyOcrOptions(
    lang=["ar", "en"],
    force_full_page_ocr=False
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)
```

### نمط تحليل المستند (80% من الملفات)

#### مقاييس المستند القياسية
```python
print(f'Total pages: {len(result.document.pages)}')
print(f'العنوان: {self.analysis.title}')
print(f'عدد المشاهد: {len(self.scenes)}')
```

#### تحليل البنية التفصيلي
```python
text_count = 0
table_count = 0
image_count = 0

for page in result.document.pages:
    text_count += len(page.texts)
    table_count += len(page.tables)
    image_count += len(page.pictures)
```

### نمط معالجة المصدر (100% من الملفات)

#### عناوين URL لـ arXiv كمصدر اختبار أساسي
```python
source = 'https://arxiv.org/pdf/2408.09869'
arabic_sources = [
    'https://arxiv.org/pdf/2310.12345',
    'https://www.aljazeera.net/documents/sample.pdf'
]
```

#### معالجة مصادر متعددة مع التعامل مع الفشل
```python
for source in arabic_sources:
    print(f'Testing with: {source}')
    try:
        result = converter.convert(source)
        # معالجة ناجحة
        break
    except Exception as e:
        print(f'Error with {source}: {e}')
        continue
```

## معايير التطوير

### بنية الدالة

#### مسؤولية واحدة واضحة (100% من الملفات)
- كل دالة لها غرض واحد محدد
- أسماء الدوال تعكس وظيفتها بدقة
- تجنب الدوال متعددة الأغراض

#### حارس Main المعياري (100% من الملفات)
```python
if __name__ == '__main__':
    main()
    # أو
    comprehensive_layout_test()
```

#### دوال بدون معاملات للاختبارات (80% من ملفات الاختبار)
```python
def comprehensive_layout_test():
    # منطق الاختبار بدون معاملات للبساطة
```

### معايير إدخال/إخراج الملفات

#### ترميز UTF-8 الصريح (100% من الملفات)
```python
with open('output.md', 'w', encoding='utf-8') as f:
    f.write(content)

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
```

#### مديرو السياق الإلزاميون (100% من الملفات)
```python
# استخدام دائم لـ with open()
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

#### أسماء ملفات وصفية تطابق الغرض
```python
'comprehensive_layout.md'
'arabic_output.md'
'docling_full_pipeline.log'
f"{stem}_dataset.json"
```

### نهج الاختبار

#### التعقيد التدريجي (100% من ملفات الاختبار)
- البدء بأمثلة أساسية (`docling_basic_example.py`)
- التقدم إلى اختبارات شاملة (`docling_comprehensive_test.py`)
- اختبارات متخصصة (`docling_arabic_rtl_test.py`)

#### الوعي بالأجهزة (60% من الملفات)
```python
# فصل نهج اختبار CPU و GPU
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4,
    device=AcceleratorDevice.AUTO
)
```

#### دعم اللغة المتخصص (40% من الملفات)
```python
# اختبار مخصص للمحتوى متعدد اللغات
ocr_options = EasyOcrOptions(
    lang=["ar", "en"],
    force_full_page_ocr=False
)
```

### أنماط التوثيق

#### التعليقات المضمنة الوصفية (80% من الملفات)
```python
# تفعيل OCR للنص العربي
pipeline_options.do_ocr = True

# إعداد المسرّع (GPU إذا متاح، وإلا CPU)
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=4,
    device=AcceleratorDevice.AUTO
)
```

#### أوصاف المخرجات الواضحة (100% من الملفات)
```python
print('=== Files Created ===')
print('- comprehensive_layout.md')
print('- comprehensive_layout.json')

logger.info("الملفات المُنشأة:")
for name, path in output_files.items():
    print(f"  • {name}: {path}")
```

#### سياق الخطأ المفيد (100% من الملفات)
```python
logger.error(f"فشل تحويل {input_path}: {e}")
logger.warning(f"الملف غير موجود: {input_path}")
print(f'Error with {source}: {e}')
```

## أنماط البرمجة المتقدمة

### استخدام Dataclasses (60% من الملفات المعقدة)
```python
@dataclass
class ConversionConfig:
    """إعدادات تحويل PDF"""
    page_width: float = 595.27
    page_height: float = 841.89
    margin_top: float = 72.0
    source_files: list = field(default_factory=list)
```

### معالجة النصوص متعددة اللغات (40% من الملفات)
```python
class ArabicTextProcessor:
    """معالج النصوص العربية للتحويل الصحيح في PDF"""
    
    def process(self, text: str) -> str:
        if any('\\u0600' <= char <= '\\u06FF' for char in text):
            # معالجة خاصة للنصوص العربية
            reshaped = self._reshaper.reshape(text)
            return self._get_display(reshaped)
        return text
```

### نمط Factory للتكوين (40% من الملفات المعقدة)
```python
def _create_pipeline_options(self) -> PdfPipelineOptions:
    """إنشاء خيارات خط الأنابيب مع تفعيل جميع القدرات"""
    pipeline_options = PdfPipelineOptions()
    # تكوين مفصل...
    return pipeline_options
```

### إدارة التبعيات الذكية (20% من الملفات)
```python
def ensure_dependencies() -> bool:
    """التحقق من وجود المكتبات المطلوبة وتثبيتها إذا لزم الأمر"""
    required_packages = {
        'reportlab': 'reportlab',
        'arabic_reshaper': 'arabic-reshaper'
    }
    # منطق التحقق والتثبيت...
```