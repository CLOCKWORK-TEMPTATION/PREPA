# Docling Simple Local File Example
from docling.document_converter import DocumentConverter

def test_local_file():
    converter = DocumentConverter()
    
    # Test with a simple approach
    try:
        # Create a simple test document path
        import os
        current_dir = os.getcwd()
        print(f'Current directory: {current_dir}')
        
        # List available files
        files = os.listdir(current_dir)
        pdf_files = [f for f in files if f.endswith('.pdf')]
        print(f'Available PDF files: {pdf_files[:5]}')  # Show first 5
        
        if pdf_files:
            source = os.path.join(current_dir, pdf_files[0])
            print(f'Testing with: {source}')
            
            result = converter.convert(source)
            print('=== Conversion successful! ===')
            print(result.document.export_to_markdown())
        else:
            print('No PDF files found. Creating a simple test...')
            # Test with a URL that should work
            source = 'https://arxiv.org/pdf/2301.07023'  # Different paper
            print(f'Testing with URL: {source}')
            
            result = converter.convert(source)
            print('=== Conversion successful! ===')
            print(result.document.export_to_markdown())
            
    except Exception as e:
        print(f'Error: {e}')
        print('This is a common issue with complex documents.')

if __name__ == '__main__':
    test_local_file()
