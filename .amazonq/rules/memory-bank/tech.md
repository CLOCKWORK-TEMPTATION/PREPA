# المجموعة التقنية

## لغة البرمجة

### Python
- **الإصدار**: Python 3.8+ (مطلوب للتوافق مع مكتبات الذكاء الاصطناعي)
- **الاستخدام**: لغة التطوير الأساسية لجميع وحدات المعالجة والتحليل
- **المبررات**: دعم ممتاز لمعالجة النصوص، مكتبات AI/ML غنية، تكامل سهل مع APIs

## التبعيات الأساسية

### معالجة المستندات

#### Docling Framework
```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
```
- **الغرض**: مكتبة تحويل المستندات الرئيسية
- **الميزات**: تحويل PDF إلى Markdown/JSON، دعم OCR، معالجة التخطيط
- **التكوين**: خيارات خط الأنابيب للتحكم في المعالجة

#### مكتبات معالجة الملفات
```python
import json          # معالجة وتصدير بيانات JSON
import csv           # تصدير البيانات الجدولية
import pathlib       # إدارة مسارات الملفات
from pathlib import Path
```

### الذكاء الاصطناعي والتحليل

#### Google Gemini AI
```python
import google.generativeai as genai
```
- **الاستخدام**: تحليل المشاعر المتقدم، تحليل الشخصيات، تحليل الحبكة
- **التكوين**: مفاتيح API في ملف .env
- **الميزات**: تحليل نصوص طويلة، دعم متعدد اللغات، تحليل سياقي

#### مكتبات التحليل النصي
```python
import re            # التعبيرات النمطية لاستخراج النصوص
import time          # قياس الأداء وإدارة معدل الطلبات
from collections import defaultdict, Counter
```

### مكتبات النظام والأداء

#### إدارة النظام
```python
import os            # متغيرات البيئة وإدارة النظام
import sys           # معلومات النظام والمسارات
import logging       # تسجيل الأحداث والأخطاء
```

#### معالجة البيانات
```python
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import asyncio       # المعالجة غير المتزامنة
```

## بيئة التطوير

### نظام التشغيل
- **المنصة الأساسية**: Windows (مسارات بصيغة `e:\PREPA\`)
- **التوافق**: متعدد المنصات مع تعديلات بسيطة للمسارات

### إدارة الملفات والترميز

#### ترميز النصوص
```python
# ترميز UTF-8 القياسي لجميع عمليات الملفات
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
```
- **الترميز القياسي**: UTF-8 لدعم النصوص متعددة اللغات
- **دعم RTL**: معالجة خاصة للنصوص العربية من اليمين إلى اليسار

#### إدارة المسارات
```python
from pathlib import Path
output_dir = Path(output_path)
output_dir.mkdir(parents=True, exist_ok=True)
```

### أوضاع المعالجة

#### معالجة CPU
- **الاستخدام**: الوضع الافتراضي والاحتياطي
- **التحسينات**: معالجة متسلسلة محسّنة للذاكرة
- **الملفات**: `docling_cpu_final.py`, `docling_simple_cpu.py`

#### معالجة GPU
- **الاستخدام**: تسريع الأجهزة عند التوفر
- **المتطلبات**: CUDA-compatible GPU
- **الملفات**: `docling_gpu_test.py`
- **التدهور التدريجي**: العودة إلى CPU عند فشل GPU

#### معالجة OCR
```python
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
converter = DocumentConverter(format_options={'pdf': pipeline_options})
```

## الموارد الخارجية

### مصادر البيانات

#### مستودع arXiv
- **URL Pattern**: `https://arxiv.org/pdf/[paper_id].pdf`
- **الاستخدام**: مصدر أساسي للمستندات الأكاديمية للاختبار
- **المميزات**: بنية مستند موحدة، وصول موثوق، محتوى متنوع

#### ملفات محلية
- **تنسيقات مدعومة**: PDF, TXT, DOC, DOCX
- **المعالجة**: تحويل تلقائي إلى تنسيقات قابلة للتحليل

### خدمات السحابة

#### Google Gemini API
```python
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')
```
- **المصادقة**: مفاتيح API في متغيرات البيئة
- **الحدود**: إدارة معدل الطلبات مع time.sleep()
- **معالجة الأخطاء**: آليات إعادة المحاولة والتدهور التدريجي

## تنسيقات البيانات

### تنسيقات الإدخال
- **PDF**: المستندات الأكاديمية والكتب
- **TXT**: النصوص العادية والسيناريوهات
- **URLs**: روابط مباشرة للمستندات عبر الإنترنت

### تنسيقات الإخراج

#### Markdown
```python
markdown_output = result.document.export_to_markdown()
```
- **الاستخدام**: عرض قابل للقراءة البشرية
- **الميزات**: الحفاظ على التنسيق والبنية

#### JSON
```python
json_output = result.document.export_to_json()
```
- **الاستخدام**: البيانات المنظمة للمعالجة الآلية
- **البنية**: هيكل هرمي يحافظ على علاقات العناصر

#### تنسيقات التدريب
```python
# Alpaca Format
{
    "instruction": "...",
    "input": "...",
    "output": "..."
}

# ShareGPT Format
{
    "conversations": [
        {"from": "system", "value": "..."},
        {"from": "user", "value": "..."}
    ]
}

# RAG Format (JSONL)
{"id": "...", "text": "...", "metadata": {...}}
```

## إدارة التكوين

### متغيرات البيئة
```bash
# .env file
GEMINI_API_KEY=your_api_key_here
LOG_LEVEL=INFO
OUTPUT_DIR=./dataset_output
```

### خيارات التكوين
```python
class Config:
    CONTEXT_WINDOW_SIZE = 5
    MAX_RETRIES = 3
    BATCH_SIZE = 10
    DEFAULT_ENCODING = 'utf-8'
```

## اعتبارات الأداء

### إدارة الذاكرة
- **معالجة دفعية**: تجنب تحميل ملفات كبيرة كاملة في الذاكرة
- **تنظيف الموارد**: استخدام context managers لإدارة الملفات
- **تحسين الاستعلامات**: تقليل عدد استدعاءات API

### قياس الأداء
```python
import time
start_time = time.time()
# كود المعالجة
processing_time = time.time() - start_time
logger.info(f'وقت المعالجة: {processing_time:.2f} ثانية')
```

### معالجة الأخطاء
```python
try:
    result = converter.convert(source)
except Exception as e:
    logger.error(f'خطأ في التحويل: {e}')
    # آليات التعافي والبدائل
```

## أدوات التطوير

### التسجيل والمراقبة
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
```

### اختبار الجودة
- **اختبارات الوحدة**: ملفات `*_test.py` للتحقق من الوظائف
- **اختبارات التكامل**: `*_comprehensive_test.py` للاختبار الشامل
- **اختبارات الأداء**: مقارنة أوقات المعالجة بين CPU/GPU

### إدارة الإصدارات
- **ملفات السجل**: تتبع مفصل لجميع العمليات
- **تسمية الملفات**: أسماء وصفية تشير إلى الإصدار والغرض
- **نسخ احتياطية**: حفظ المخرجات في دلائل منفصلة