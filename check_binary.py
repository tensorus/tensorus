try:
    with open('tensorus/tensor_storage.py', 'rb') as f:
        content = f.read()
        print(f'File size: {len(content)} bytes')
        
        # Try to decode with different encodings
        try:
            text = content.decode('utf-8')
            lines = text.split('\n')
            print(f'UTF-8 decoding successful, {len(lines)} lines')
            
            # Check lines around 510
            start = max(0, 508)
            end = min(len(lines), 512)
            for i in range(start, end):
                print(f'Line {i+1}: {repr(lines[i])}')
        except UnicodeDecodeError as e:
            print(f'UTF-8 decoding failed: {e}')
            
            # Try with error handling
            text = content.decode('utf-8', errors='replace')
            lines = text.split('\n')
            print(f'UTF-8 decoding with replacement, {len(lines)} lines')
            
            # Check lines around 510
            start = max(0, 508)
            end = min(len(lines), 512)
            for i in range(start, end):
                print(f'Line {i+1}: {repr(lines[i])}')
except Exception as e:
    print(f'Error reading file: {type(e).__name__}: {e}')
