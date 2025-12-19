# وثيقة تصميم نظام الراوي الإصدار 4.0

## نظرة عامة

يهدف الإصدار 4.0 من نظام الراوي إلى تحسين جودة مجموعات البيانات المستخرجة من السيناريوهات العربية من خلال أربع تحسينات هندسية رئيسية:

1. **توحيد الكيانات**: دمج أسماء الشخصيات المتشابهة لتحقيق اتساق البيانات
2. **إثراء السياق**: إضافة الأوصاف السردية إلى بيانات التدريب
3. **فلترة الجودة**: إزالة الحوارات منخفضة القيمة المعلوماتية تلقائياً
4. **الميتاداتا الزمنية**: استخراج وتوثيق الفترات الزمنية للمشاهد

تم تصميم هذه التحسينات للتكامل السلس مع البنية المعمارية الحالية دون إعادة كتابة شاملة للنظام.

## البنية المعمارية

### البنية الحالية

النظام الحالي يتبع نمط Pipeline Architecture مع أربع مراحل رئيسية:

```
Ingestor → Parser → Enricher → Exporter
```

1. **Ingestor**: قراءة الملفات (TXT/PDF) وتحويلها إلى نص خام
2. **Parser**: تحليل النص وتقسيمه إلى مشاهد وحوارات
3. **Enricher**: إضافة التضمينات وتحليل المشاعر
4. **Exporter**: تصدير البيانات بصيغ متعددة (Alpaca, ShareGPT, JSONL, CSV)

### التحسينات المعمارية في v4.0

سيتم دمج التحسينات الأربعة في المراحل المناسبة:

```
┌─────────────────────────────────────────────────────────────┐
│                        Ingestor                              │
│                    (بدون تغييرات)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                        Parser                                │
│  + استخراج الميتاداتا الزمنية (time_period)                │
│  + تحسين استخراج الأسطر الوصفية (action lines)             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                       Enricher                               │
│  + بناء قاموس توحيد الكيانات (Entity Canonicalization)     │
│  + تطبيق التطبيع على جميع الحوارات                         │
│  (تحليل المشاعر موجود بالفعل)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                       Exporter                               │
│  + فلترة الجودة التلقائية                                  │
│  + إثراء السياق في تصدير Alpaca                            │
│  + إضافة time_period إلى JSONL                             │
└─────────────────────────────────────────────────────────────┘
```

## المكونات والواجهات

### 1. وحدة توحيد الكيانات (Entity Canonicalizer)

#### الموقع
سيتم إضافتها كفئة جديدة `EntityCanonicalizer` في طبقة Enricher.

#### الواجهة

```python
class EntityCanonicalizer:
    """
    مسؤول عن توحيد أسماء الشخصيات المتشابهة
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: نسبة التشابه المطلوبة للدمج (افتراضي: 85%)
        """
        self.threshold = similarity_threshold
        self.canonical_map: Dict[str, str] = {}
        self.merge_log: List[Dict[str, Any]] = []
    
    def build_canonical_map(self, scenes: List[Scene]) -> Dict[str, str]:
        """
        بناء قاموس التطبيع من جميع المشاهد
        
        Returns:
            قاموس يربط الأسماء المتشابهة بالاسم الكانوني
        """
        pass
    
    def normalize_character_name(self, name: str) -> str:
        """
        تطبيع اسم شخصية واحدة
        
        Returns:
            الاسم الكانوني
        """
        pass
    
    def apply_normalization(self, scenes: List[Scene]) -> List[Scene]:
        """
        تطبيق التطبيع على جميع الحوارات في المشاهد
        """
        pass
    
    def export_merge_log(self, output_path: Path):
        """
        حفظ سجل عمليات الدمج
        """
        pass
```

#### الخوارزمية

1. **جمع جميع الأسماء**: استخراج قائمة بجميع أسماء الشخصيات من كل المشاهد
2. **حساب التكرارات**: عد مرات ظهور كل اسم
3. **حساب التشابه**: لكل زوج من الأسماء، حساب نسبة التشابه باستخدام:
   - `rapidfuzz.fuzz.ratio()` (الخيار الأول - أسرع)
   - `difflib.SequenceMatcher().ratio()` (الخيار البديل)
4. **بناء القاموس**: إذا كانت نسبة التشابه > 85%:
   - اختيار الاسم الأكثر تكراراً كاسم كانوني
   - إذا تساوت التكرارات، اختيار الاسم الأطول
5. **التطبيق**: استبدال جميع الأسماء في DialogueTurn.speaker

#### مثال على التطبيع

```
الأسماء الأصلية:
- "رأفت" (50 مرة)
- "رأفت عبد المجيد" (30 مرة)
- "رأفت الهجان" (10 مرات)

نسبة التشابه:
- "رأفت" ↔ "رأفت عبد المجيد": 0.87 (> 0.85)
- "رأفت" ↔ "رأفت الهجان": 0.89 (> 0.85)

النتيجة:
- الاسم الكانوني: "رأفت عبد المجيد" (الأطول من الأكثر تكراراً)
- التطبيع: جميع الأسماء → "رأفت عبد المجيد"
```

### 2. وحدة إثراء السياق (Context Enricher)

#### الموقع
تعديل على الدالة `export_contextual_alpaca` في فئة `DatasetExporter`.

#### التعديلات المطلوبة

```python
def export_contextual_alpaca(self, scenes: List[Scene]):
    """
    تصدير بصيغة Alpaca مع سياق محسّن
    """
    data = []
    for scene in scenes:
        dialogue = scene.dialogue
        if not dialogue: continue

        # استخراج آخر سطر وصفي مهم
        last_action = self._get_last_significant_action(scene.actions)
        
        # بناء وصف المشهد المحسّن
        scene_setup = self._build_enriched_scene_setup(
            scene, 
            last_action
        )
        
        context_buffer = []
        
        for i, turn in enumerate(dialogue):
            # إضافة السياق الوصفي قبل كل حوار
            action_context = self._get_action_before_turn(
                scene.actions, 
                i
            )
            
            current_history = "\n".join(context_buffer) if context_buffer else "بداية الحوار."
            
            full_input = f"{scene_setup}"
            if action_context:
                full_input += f"\n[سياق: {action_context}]"
            full_input += f"\n\nسياق الحديث السابق:\n{current_history}\n\nالمتحدث: {turn.speaker}"
            
            # ... باقي الكود
```

#### الدوال المساعدة الجديدة

```python
def _get_last_significant_action(self, actions: List[str]) -> str:
    """
    استخراج آخر سطر وصفي مهم (أكثر من 10 أحرف، ليس انتقال)
    """
    pass

def _build_enriched_scene_setup(self, scene: Scene, last_action: str) -> str:
    """
    بناء وصف المشهد مع السياق الوصفي
    """
    pass

def _get_action_before_turn(self, actions: List[str], turn_index: int) -> str:
    """
    استخراج السطر الوصفي الذي يسبق حوار معين
    (يتطلب تتبع موضع الحوارات في النص الأصلي)
    """
    pass
```

### 3. وحدة فلترة الجودة (Quality Filter)

#### الموقع
دالة جديدة في فئة `DatasetExporter` تُطبق قبل التصدير.

#### الواجهة

```python
class QualityFilter:
    """
    فلترة الحوارات منخفضة الجودة
    """
    
    def __init__(self, min_words: int = 3, high_sentiment_threshold: float = 0.8):
        self.min_words = min_words
        self.sentiment_threshold = high_sentiment_threshold
        self.filtered_count = 0
    
    def should_keep_turn(self, turn: DialogueTurn) -> bool:
        """
        تحديد ما إذا كان يجب الاحتفاظ بالحوار
        
        Returns:
            True إذا كان الحوار عالي الجودة
        """
        word_count = count_arabic_words(turn.text)
        
        # قاعدة 1: الحوارات الطويلة تُحفظ دائماً
        if word_count >= self.min_words:
            return True
        
        # قاعدة 2: الحوارات القصيرة ذات المشاعر القوية تُحفظ
        if turn.sentiment_score >= self.sentiment_threshold:
            return True
        
        # خلاف ذلك، تُحذف
        return False
    
    def filter_scenes(self, scenes: List[Scene]) -> List[Scene]:
        """
        تطبيق الفلترة على جميع المشاهد
        """
        filtered_scenes = []
        for scene in scenes:
            filtered_dialogue = [
                turn for turn in scene.dialogue 
                if self.should_keep_turn(turn)
            ]
            self.filtered_count += len(scene.dialogue) - len(filtered_dialogue)
            
            # إنشاء نسخة جديدة من المشهد مع الحوارات المفلترة
            filtered_scene = Scene(
                scene_id=scene.scene_id,
                scene_number=scene.scene_number,
                heading=scene.heading,
                location=scene.location,
                time_of_day=scene.time_of_day,
                int_ext=scene.int_ext,
                actions=scene.actions,
                dialogue=filtered_dialogue,
                characters=list(set(t.speaker for t in filtered_dialogue)),
                full_text=scene.full_text,
                embedding=scene.embedding
            )
            filtered_scenes.append(filtered_scene)
        
        return filtered_scenes
    
    def get_filter_stats(self) -> Dict[str, int]:
        """
        الحصول على إحصائيات الفلترة
        """
        return {
            "filtered_count": self.filtered_count
        }
```

### 4. وحدة استخراج الميتاداتا الزمنية (Temporal Metadata Extractor)

#### الموقع
تعديل على فئة `Scene` وفئة `ScreenplayParser`.

#### تعديل نموذج البيانات

```python
@dataclass
class Scene:
    scene_id: str
    scene_number: int
    heading: str
    location: str
    time_of_day: str
    int_ext: str
    time_period: str = "غير محدد"  # حقل جديد
    actions: List[str] = field(default_factory=list)
    dialogue: List[DialogueTurn] = field(default_factory=list)
    characters: List[str] = field(default_factory=list)
    full_text: str = ""
    embedding: Optional[List[float]] = None
```

#### التعديلات على Parser

```python
class ScreenplayParser:
    # إضافة نمط regex للسنوات
    YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
    
    def __init__(self):
        self.normalizer = self
        self.last_known_year = "غير محدد"  # لتتبع السنة عبر المشاهد
    
    def _extract_time_period(self, scene_text: str) -> str:
        """
        استخراج الفترة الزمنية من نص المشهد
        
        Args:
            scene_text: نص عنوان المشهد أو محتواه
            
        Returns:
            السنة إن وُجدت، أو "غير محدد"
        """
        match = self.YEAR_PATTERN.search(scene_text)
        if match:
            year = match.group(0)
            self.last_known_year = year
            return year
        return self.last_known_year
    
    def parse(self, lines: List[str]) -> List[Scene]:
        # ... الكود الموجود
        
        # عند إنشاء مشهد جديد:
        current_scene = Scene(
            scene_id=f"S{num:04d}",
            scene_number=num,
            heading=line,
            location=loc_val or "موقع غير محدد",
            time_of_day=time_val,
            int_ext="داخلي" if "داخلي" in line else "خارجي",
            time_period=self._extract_time_period(line)  # جديد
        )
        
        # ... باقي الكود
```

## نماذج البيانات

### التعديلات على النماذج الموجودة

```python
@dataclass
class DialogueTurn:
    scene_id: str
    turn_id: int
    speaker: str  # سيتم تطبيعه بواسطة EntityCanonicalizer
    text: str
    normalized_text: str = ""
    sentiment: str = "unknown"
    sentiment_score: float = 0.0
    original_speaker: str = ""  # جديد: للاحتفاظ بالاسم الأصلي

@dataclass
class Scene:
    scene_id: str
    scene_number: int
    heading: str
    location: str
    time_of_day: str
    int_ext: str
    time_period: str = "غير محدد"  # جديد
    actions: List[str] = field(default_factory=list)
    dialogue: List[DialogueTurn] = field(default_factory=list)
    characters: List[str] = field(default_factory=list)
    full_text: str = ""
    embedding: Optional[List[float]] = None
```

## خصائص الصحة (Correctness Properties)

*خاصية هي سمة أو سلوك يجب أن يكون صحيحاً عبر جميع عمليات التنفيذ الصالحة للنظام - في الأساس، بيان رسمي حول ما يجب أن يفعله النظام. الخصائص تعمل كجسر بين المواصفات المقروءة للبشر وضمانات الصحة القابلة للتحقق آلياً.*

### خصائص توحيد الكيانات

**خاصية 1: حساب التشابه للأسماء المتشابهة**
*لأي* مجموعة من أسماء الشخصيات، عندما يواجه النظام أسماء متشابهة، يجب أن يحسب المسافة الليفنشتاين بينها باستخدام rapidfuzz أو difflib
**تتحقق من: المتطلبات 1.1, 1.5**

**خاصية 2: ربط الأسماء عالية التشابه**
*لأي* زوج من الأسماء بنسبة تشابه أعلى من 85%، يجب أن يربط النظام الاسم الأقصر بالاسم الأكثر تكراراً كاسم كانوني
**تتحقق من: المتطلبات 1.2**

**خاصية 3: تطبيق التطبيع الشامل**
*لأي* مجموعة من المشاهد مع قاموس تطبيع، يجب أن يطبق النظام التطبيع على جميع كائنات DialogueTurn قبل التصدير
**تتحقق من: المتطلبات 1.3**

**خاصية 4: توثيق عمليات الدمج**
*لأي* عملية تطبيع تحدث، يجب أن يحتفظ النظام بسجل للأسماء المدمجة في ملف منفصل
**تتحقق من: المتطلبات 1.4**

### خصائص إثراء السياق

**خاصية 5: تضمين السياق الوصفي**
*لأي* عملية تصدير Alpaca، يجب أن يتضمن النظام آخر سطر وصفي مهم قبل الحوار في حقل الإدخال
**تتحقق من: المتطلبات 2.1**

**خاصية 6: دمج معلومات المشهد**
*لأي* حوار يتم تنسيقه، يجب أن يدمج النظام معلومات المكان والزمان مع السياق الوصفي في حقل الإدخال
**تتحقق من: المتطلبات 2.2**

**خاصية 7: استخراج الأسطر الوصفية**
*لأي* مشهد يحتوي على قائمة actions، يجب أن يستخرج النظام الأسطر الوصفية منها لاستخدامها في السياق
**تتحقق من: المتطلبات 2.3**

**خاصية 8: تنسيق الإدخال المعياري**
*لأي* حوار يتم تنسيقه، يجب أن يستخدم النظام التنسيق "المكان: X. [سياق: Y]. المتحدث: Z"
**تتحقق من: المتطلبات 2.4**

### خصائص فلترة الجودة

**خاصية 9: فلترة الحوارات القصيرة**
*لأي* حوار يحتوي على أقل من 3 كلمات عربية، يجب أن يحذفه النظام ما لم تكن درجة المشاعر أعلى من 0.8
**تتحقق من: المتطلبات 3.1, 3.2**

**خاصية 10: الاحتفاظ بالحوارات العاطفية**
*لأي* حوار قصير بدرجة مشاعر أعلى من 0.8، يجب أن يحتفظ به النظام رغم قصره
**تتحقق من: المتطلبات 3.2**

**خاصية 11: تسجيل إحصائيات الفلترة**
*لأي* عملية فلترة تحدث، يجب أن يسجل النظام عدد الحوارات المحذوفة في ملف السجل
**تتحقق من: المتطلبات 3.3**

**خاصية 12: تطبيق الفلترة قبل التصدير**
*لأي* عملية تصدير، يجب أن تحدث الفلترة في مرحلة التصدير قبل إنشاء الملفات
**تتحقق من: المتطلبات 3.5**

### خصائص الميتاداتا الزمنية

**خاصية 13: استخراج السنوات من العناوين**
*لأي* عنوان مشهد، يجب أن يبحث النظام عن السنوات باستخدام النمط \b(19|20)\d{2}\b ويستخرجها
**تتحقق من: المتطلبات 4.1**

**خاصية 14: إضافة حقل الفترة الزمنية**
*لأي* مشهد يتم العثور على سنة في نصه، يجب أن يضيف النظام حقل time_period إلى كائن Scene
**تتحقق من: المتطلبات 4.2**

**خاصية 15: وراثة الفترة الزمنية**
*لأي* مشهد لا يحتوي على سنة، يجب أن يرث النظام السنة من المشهد السابق أو يضع "غير محدد"
**تتحقق من: المتطلبات 4.3**

**خاصية 16: تضمين الفترة في التصدير**
*لأي* عملية تصدير JSONL، يجب أن يتضمن النظام حقل time_period في الميتاداتا
**تتحقق من: المتطلبات 4.4**

**خاصية 17: البحث في محتوى المشهد**
*لأي* مشهد يتم معالجته، يجب أن يبحث النظام عن المؤشرات الزمنية في محتوى المشهد وليس فقط العنوان
**تتحقق من: المتطلبات 4.5**

### خصائص التكامل والموثوقية

**خاصية 18: معالجة فشل المكتبات**
*لأي* مكتبة خارجية تفشل في الاستيراد، يجب أن يتعامل النظام مع الاستيراد بطريقة آمنة ويتابع العمل بدون التحسين المعتمد عليها
**تتحقق من: المتطلبات 5.2, 6.3**

**خاصية 19: إنتاج الملفات المتوقعة**
*لأي* عملية تشغيل للنظام، يجب أن ينتج نفس أنواع الملفات المخرجة مع البيانات المحسنة
**تتحقق من: المتطلبات 5.5**

**خاصية 20: تسجيل الأخطاء**
*لأي* خطأ يحدث في أي مرحلة، يجب أن يسجل النظام رسالة خطأ واضحة باللغة العربية
**تتحقق من: المتطلبات 6.1**

**خاصية 21: تسجيل الإحصائيات**
*لأي* تحسين يتم تطبيقه، يجب أن يسجل النظام إحصائيات العمليات المنجزة
**تتحقق من: المتطلبات 6.2**

**خاصية 22: التحقق من البيانات**
*لأي* ملف يتم معالجته، يجب أن يتحقق النظام من وجود البيانات المطلوبة قبل المعالجة
**تتحقق من: المتطلبات 6.4**

**خاصية 23: ضمان نجاح الكتابة**
*لأي* ملف يتم حفظه، يجب أن يتأكد النظام من نجاح عملية الكتابة قبل المتابعة
**تتحقق من: المتطلبات 6.5**


## معالجة الأخطاء

### استراتيجية معالجة الأخطاء

النظام يتبع مبدأ "Graceful Degradation" - إذا فشل أحد التحسينات، يجب أن يتابع النظام العمل بدون ذلك التحسين:

#### 1. فشل مكتبات التشابه

```python
try:
    import rapidfuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    try:
        import difflib
        DIFFLIB_AVAILABLE = True
    except ImportError:
        DIFFLIB_AVAILABLE = False
        logger.warning("لا توجد مكتبات حساب التشابه - تخطي توحيد الكيانات")
```

#### 2. فشل تحليل المشاعر

إذا فشل نموذج تحليل المشاعر، يجب أن تعمل فلترة الجودة بناءً على عدد الكلمات فقط:

```python
def should_keep_turn(self, turn: DialogueTurn) -> bool:
    word_count = count_arabic_words(turn.text)
    
    if word_count >= self.min_words:
        return True
    
    # إذا لم يكن تحليل المشاعر متوفراً، احتفظ بالحوارات القصيرة
    if turn.sentiment == "unknown":
        logger.warning("تحليل المشاعر غير متوفر - الاحتفاظ بالحوار القصير")
        return True
    
    return turn.sentiment_score >= self.sentiment_threshold
```

#### 3. فشل استخراج الميتاداتا

إذا فشل استخراج السنوات، يجب وضع "غير محدد" والمتابعة:

```python
def _extract_time_period(self, scene_text: str) -> str:
    try:
        match = self.YEAR_PATTERN.search(scene_text)
        if match:
            year = match.group(0)
            self.last_known_year = year
            return year
        return self.last_known_year
    except Exception as e:
        logger.error(f"فشل استخراج الفترة الزمنية: {e}")
        return "غير محدد"
```

### تسجيل الأخطاء

جميع الأخطاء يجب أن تُسجل بوضوح مع معلومات السياق:

```python
logger.error(f"فشل في توحيد الكيان '{original_name}': {str(e)}")
logger.warning(f"تخطي إثراء السياق للمشهد {scene.scene_id}: لا توجد أسطر وصفية")
logger.info(f"تم فلترة {filtered_count} حوار من أصل {total_count}")
```

## استراتيجية الاختبار

### الاختبار المزدوج

النظام يتطلب نهج اختبار مزدوج يجمع بين:

1. **اختبارات الوحدة (Unit Tests)**: للتحقق من أمثلة محددة وحالات الحافة
2. **اختبارات الخصائص (Property-Based Tests)**: للتحقق من الخصائص العامة عبر مدخلات متنوعة

### مكتبة الاختبار المختارة

سيتم استخدام **Hypothesis** لاختبارات الخصائص في Python:

```python
pip install hypothesis
```

### متطلبات اختبارات الخصائص

- كل اختبار خاصية يجب أن يعمل لـ 100 تكرار على الأقل
- كل اختبار يجب أن يحتوي على تعليق يربطه بالخاصية في وثيقة التصميم
- التنسيق المطلوب: `**Feature: al-rawi-v4, Property {number}: {property_text}**`

### أمثلة على اختبارات الخصائص

```python
from hypothesis import given, strategies as st
import hypothesis.strategies as st

@given(st.lists(st.text(min_size=2, max_size=20), min_size=2, max_size=10))
def test_entity_canonicalization_similarity_calculation(character_names):
    """
    **Feature: al-rawi-v4, Property 1: حساب التشابه للأسماء المتشابهة**
    """
    canonicalizer = EntityCanonicalizer()
    # اختبار أن النظام يحسب التشابه لأي مجموعة أسماء
    canonical_map = canonicalizer.build_canonical_map_from_names(character_names)
    # التحقق من أن جميع الأسماء المتشابهة تم ربطها
    assert isinstance(canonical_map, dict)

@given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20))
def test_quality_filter_short_dialogues(dialogue_texts):
    """
    **Feature: al-rawi-v4, Property 9: فلترة الحوارات القصيرة**
    """
    filter = QualityFilter(min_words=3)
    for text in dialogue_texts:
        turn = DialogueTurn(
            scene_id="test", turn_id=1, speaker="test", 
            text=text, sentiment_score=0.5
        )
        result = filter.should_keep_turn(turn)
        word_count = count_arabic_words(text)
        
        if word_count >= 3:
            assert result == True
        elif turn.sentiment_score >= 0.8:
            assert result == True
        else:
            assert result == False
```

### اختبارات الوحدة

اختبارات الوحدة تركز على حالات محددة:

```python
def test_time_period_extraction_specific_years():
    """اختبار استخراج سنوات محددة"""
    parser = ScreenplayParser()
    
    # حالات محددة
    assert parser._extract_time_period("مشهد 1 - 1986") == "1986"
    assert parser._extract_time_period("داخلي - منزل - 2009") == "2009"
    assert parser._extract_time_period("خارجي - شارع") == "غير محدد"

def test_context_enrichment_no_actions():
    """اختبار إثراء السياق عند عدم وجود أسطر وصفية"""
    exporter = DatasetExporter("test_output")
    scene = Scene(
        scene_id="S001", scene_number=1, heading="مشهد 1",
        location="غرفة", time_of_day="نهار", int_ext="داخلي",
        actions=[], dialogue=[]
    )
    
    # يجب أن يستخدم عنوان المشهد فقط
    setup = exporter._build_enriched_scene_setup(scene, "")
    assert "مشهد 1" in setup
    assert "غرفة" in setup
```

## الاعتبارات الأمنية

### حماية البيانات

1. **تشفير الملفات الحساسة**: إذا كانت السيناريوهات تحتوي على محتوى حساس
2. **التحقق من صحة المدخلات**: منع هجمات حقن الكود عبر أسماء الملفات
3. **إدارة الذاكرة**: تجنب تحميل ملفات كبيرة جداً في الذاكرة

### التحقق من المدخلات

```python
def validate_input_file(file_path: str) -> bool:
    """التحقق من صحة ملف الإدخال"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"الملف غير موجود: {file_path}")
    
    if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
        raise ValueError("حجم الملف كبير جداً")
    
    allowed_extensions = ['.txt', '.pdf', '.md']
    if not any(file_path.lower().endswith(ext) for ext in allowed_extensions):
        raise ValueError("نوع الملف غير مدعوم")
    
    return True
```

## اعتبارات الأداء

### تحسين الأداء

1. **معالجة متوازية**: استخدام multiprocessing لمعالجة ملفات متعددة
2. **تخزين مؤقت**: حفظ نتائج حساب التشابه لتجنب إعادة الحساب
3. **معالجة تدفقية**: معالجة الملفات الكبيرة على دفعات

### إدارة الذاكرة

```python
def process_large_file_in_chunks(file_path: str, chunk_size: int = 1000):
    """معالجة الملفات الكبيرة على دفعات"""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            lines = []
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            
            if not lines:
                break
                
            # معالجة الدفعة
            yield lines
```

## خطة النشر

### مراحل التطوير

1. **المرحلة 1**: تطوير وحدة توحيد الكيانات
2. **المرحلة 2**: تطوير وحدة إثراء السياق
3. **المرحلة 3**: تطوير وحدة فلترة الجودة
4. **المرحلة 4**: تطوير وحدة الميتاداتا الزمنية
5. **المرحلة 5**: التكامل والاختبار الشامل

### متطلبات النشر

```python
# المكتبات الجديدة المطلوبة
requirements_v4 = [
    "rapidfuzz>=3.0.0",  # للتشابه السريع
    "hypothesis>=6.0.0",  # لاختبارات الخصائص
    # المكتبات الموجودة
    "transformers>=4.20.0",
    "sentence-transformers>=2.2.0",
    "docling>=1.0.0",
    "google-genai>=0.3.0",
    "pandas>=1.5.0",
    "networkx>=2.8.0"
]
```

### اختبار التوافق

قبل النشر، يجب اختبار النظام مع:
- ملفات سيناريو مختلفة الأحجام
- ملفات PDF وTXT
- بيئات مع وبدون GPU
- إصدارات مختلفة من Python (3.8+)

## الخلاصة

الإصدار 4.0 من نظام الراوي يقدم تحسينات جوهرية لجودة البيانات المستخرجة مع الحفاظ على البنية المعمارية المستقرة. التصميم يضمن المرونة والموثوقية من خلال معالجة الأخطاء الشاملة واستراتيجية الاختبار المزدوجة.