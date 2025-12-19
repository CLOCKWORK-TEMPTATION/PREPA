# Docling Arabic RTL Support Test
from docling.document_converter import DocumentConverter

def test_arabic_rtl():
    converter = DocumentConverter()
    
    # Test with Arabic content URL
    arabic_sources = [
        'https://arxiv.org/pdf/2310.12345',  # Example Arabic paper
        'https://www.aljazeera.net/documents/sample.pdf'  # Example Arabic news
    ]
    
    for source in arabic_sources:
        print(f'Testing with: {source}')
        try:
            result = converter.convert(source)
            markdown_output = result.document.export_to_markdown()
            
            print('=== Arabic RTL Test Results ===')
            print(markdown_output[:1000])  # First 1000 chars
            
            # Save to file
            with open('arabic_output.md', 'w', encoding='utf-8') as f:
                f.write(markdown_output)
            
            print('Arabic content saved to arabic_output.md')
            break
            
        except Exception as e:
            print(f'Error with {source}: {e}')
            continue

if __name__ == '__main__':
    test_arabic_rtl()
