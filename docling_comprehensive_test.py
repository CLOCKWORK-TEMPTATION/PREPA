# Docling Fixed Comprehensive Layout Example
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import json

def comprehensive_layout_test():
    # Create converter with default options
    converter = DocumentConverter()
    
    # Test document
    source = 'https://arxiv.org/pdf/2408.09869'
    
    try:
        print('=== Starting Comprehensive Layout Test ===')
        result = converter.convert(source)
        
        # Export to different formats
        markdown_output = result.document.export_to_markdown()
        json_output = result.document.export_to_json()
        
        # Save results
        with open('comprehensive_layout.md', 'w', encoding='utf-8') as f:
            f.write(markdown_output)
        
        with open('comprehensive_layout.json', 'w', encoding='utf-8') as f:
            f.write(json_output)
        
        print('=== Layout Analysis ===')
        print(f'Total pages: {len(result.document.pages)}')
        
        # Analyze document structure
        text_count = 0
        table_count = 0
        image_count = 0
        
        for page in result.document.pages:
            text_count += len(page.texts)
            table_count += len(page.tables)  
            image_count += len(page.pictures)
        
        print(f'Total text blocks: {text_count}')
        print(f'Total tables: {table_count}')
        print(f'Total images: {image_count}')
        
        print('\\n=== Sample Layout Output ===')
        print(markdown_output[:2000])
        
        print('\\n=== Files Created ===')
        print('- comprehensive_layout.md')
        print('- comprehensive_layout.json')
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    comprehensive_layout_test()
