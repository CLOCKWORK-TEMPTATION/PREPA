# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Arabic document processing and dataset building toolkit. It converts Arabic screenplays and PDF documents into high-quality training datasets for LLMs. The project combines Docling for document processing with custom screenplay analysis tools for sentiment analysis, character development, and dialogue extraction.

## Key Commands

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run document processing pipeline
python docling_full_pipeline.py

# Build dataset from screenplay
python script_dataset_builder.py

# Convert TXT files to PDF
python txt_to_pdf_converter.py

# Test Arabic RTL support
python docling_arabic_rtl_test.py

# CPU-optimized processing
python docling_cpu_final.py
```

## Architecture

### Processing Pipeline
```
Ingestor → Parser → Enricher → Exporter
```

1. **Document Ingestion**: PDF/TXT files loaded via Docling or direct file reading
2. **Parsing**: `ScreenplayParser` extracts scenes, dialogues, characters from Arabic text
3. **Enrichment**: `AIEnricher` adds sentiment analysis via Google Gemini API
4. **Export**: `DatasetExporter` outputs to Alpaca, ShareGPT, RAG (JSONL), CSV formats

### Core Components

- **docling_full_pipeline.py**: Main pipeline using Docling with OCR, TableFormer, Layout Analysis
- **script_dataset_builder.py**: Standalone screenplay parser without Docling dependency
- **screenplay_to_dataset.py**: Alternative processor using `unstructured` library
- **txt_to_pdf_converter.py**: Converts TXT to PDF with Arabic/RTL support

### Data Models

```python
@dataclass
class Scene:
    scene_number: int
    time_of_day: str      # ليل/نهار
    location_type: str    # داخلي/خارجي
    location_name: str
    dialogues: List[Dict[str, str]]
    characters: List[str]
    stage_directions: List[str]

@dataclass
class DialogueTurn:
    speaker: str
    text: str
    sentiment_score: float
```

### Output Formats

- **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
- **ShareGPT**: `{"conversations": [{"from": "user/assistant", "value": "..."}]}`
- **RAG/JSONL**: `{"id": "...", "text": "...", "metadata": {...}}`

## Development Standards

### Language & Communication
- Code comments and logs in Arabic
- Professional logging with `logging` module (no `print` statements)
- UTF-8 encoding for all file operations

### Code Patterns

```python
# Standard imports order
import os, re, json, time, logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Required: context managers for files
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Required: comprehensive error handling
try:
    result = converter.convert(source)
except Exception as e:
    logger.error(f"خطأ في التحويل: {e}")
```

### Docling Configuration

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = EasyOcrOptions(lang=["ar", "en"])
pipeline_options.do_table_structure = True
```

## Directory Structure

```
PREPA/
├── dataset_output/           # Main output: Alpaca, ShareGPT, RAG datasets
├── Extracted_Dataset/        # Processed data and converted PDFs
│   ├── dataset_output/
│   ├── docling_dataset_output/
│   └── pddf/                 # Converted PDF files
├── docling_*.py              # Document processing modules
├── screenplay_*.py           # Screenplay analysis modules
└── *.log                     # Processing logs
```

## Environment

- **Python**: 3.8+
- **Platform**: Windows (paths use `e:\PREPA\`)
- **Key Dependencies**: docling, google-generativeai, reportlab, arabic-reshaper

### API Configuration

```bash
# .env file
GEMINI_API_KEY=your_api_key_here
```

## Al-Rawi System (v4.0)

The "الراوي" (Al-Rawi/Narrator) system is the core screenplay processing engine with these planned enhancements:

1. **Entity Canonicalization**: Merge similar character names using Levenshtein distance (>85% similarity)
2. **Scene Context Enrichment**: Add descriptive context to Alpaca training data
3. **Quality Filtering**: Remove dialogues <3 words (unless high emotion score >0.8)
4. **Temporal Metadata**: Extract year patterns `\b(19|20)\d{2}\b` for flashback detection
