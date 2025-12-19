# -*- coding: utf-8 -*-
"""
إعدادات pytest لنظام الراوي
"""
import sys
from pathlib import Path

# إضافة المسار الجذري للمشروع
sys.path.insert(0, str(Path(__file__).parent))

# إعدادات Hypothesis
from hypothesis import settings, Verbosity

settings.register_profile("dev", max_examples=50)
settings.register_profile("ci", max_examples=100)
settings.register_profile("exhaustive", max_examples=500)
settings.load_profile("dev")
