# Docling Final Working Example
from docling.document_converter import DocumentConverter

def final_docling_demo():
    converter = DocumentConverter()
    
    # Test document
    source = 'https://arxiv.org/pdf/2408.09869'
    
    try:
        print('=== Docling Final Demo ===')
        result = converter.convert(source)
        
        # Export to markdown
        markdown_output = result.document.export_to_markdown()
        
        # Save result
        with open('docling_final_output.md', 'w', encoding='utf-8') as f:
            f.write(markdown_output)
        
        print('=== Document Analysis ===')
        print(f'Total pages: {len(result.document.pages)}')
        
        # Count elements
        total_text = sum(len(page.texts) for page in result.document.pages)
        total_tables = sum(len(page.tables) for page in result.document.pages)
        total_images = sum(len(page.pictures) for page in result.document.pages)
        
        print(f'Text blocks: {total_text}')
        print(f'Tables: {total_tables}')
        print(f'Images: {total_images}')
        
        print('\\n=== Preview ===')
        print(markdown_output[:1500])
        print('\\n=== Success! File saved as docling_final_output.md ===')
        
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    final_docling_demo()
