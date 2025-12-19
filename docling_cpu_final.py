# Docling CPU Force Test
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions

def test_docling_cpu():
    # Force CPU usage
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    
    converter = DocumentConverter(format_options={
        'pdf': pipeline_options
    })
    
    source = 'https://arxiv.org/pdf/2301.07023'
    
    try:
        print('=== Testing Docling with CPU (GPU not compatible) ===')
        import time
        start_time = time.time()
        
        result = converter.convert(source)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f'Processing time: {processing_time:.2f} seconds')
        print(f'Pages: {len(result.document.pages)}')
        
        # Save result
        with open('docling_cpu_final.md', 'w', encoding='utf-8') as f:
            f.write(result.document.export_to_markdown())
        
        print('=== CPU Test Complete! ===')
        print('File saved: docling_cpu_final.md')
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    test_docling_cpu()
