# Docling Simple CPU Test
from docling.document_converter import DocumentConverter

def test_simple_cpu():
    converter = DocumentConverter()
    source = 'https://arxiv.org/pdf/2301.07023'
    
    try:
        print('=== Testing Docling (CPU mode) ===')
        import time
        start_time = time.time()
        
        result = converter.convert(source)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f'Processing time: {processing_time:.2f} seconds')
        print(f'Pages: {len(result.document.pages)}')
        
        # Save result
        with open('docling_simple_cpu.md', 'w', encoding='utf-8') as f:
            f.write(result.document.export_to_markdown())
        
        print('=== Success! ===')
        print('File saved: docling_simple_cpu.md')
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    test_simple_cpu()
