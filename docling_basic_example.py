# Docling Basic Example for Arabic Support
from docling.document_converter import DocumentConverter

def convert_document_example():
    converter = DocumentConverter()
    
    # Can be changed to local file path
    source = 'https://arxiv.org/pdf/2408.09869'
    
    try:
        result = converter.convert(source)
        
        print('=== Converted Document (Markdown) ===')
        print(result.document.export_to_markdown())
        
        with open('output.md', 'w', encoding='utf-8') as f:
            f.write(result.document.export_to_markdown())
        
        print('\n=== Result saved to output.md ===')
        
        print(f'\n=== Document Info ===')
        print(f'Pages: {len(result.document.pages)}')
        print(f'Title: {result.document.title or "Not specified"}')
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    convert_document_example()
