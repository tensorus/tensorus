import sys

def check_file():
    try:
        with open('tensorus/tensor_storage.py', 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            print(f'Total lines: {len(lines)}')
            
            # Check lines around 510
            start = max(0, 508)
            end = min(len(lines), 512)
            for i in range(start, end):
                print(f'Line {i+1}: {repr(lines[i])}')
                
    except FileNotFoundError:
        print('File not found')
    except Exception as e:
        print(f'Error reading file: {type(e).__name__}: {e}')
        
if __name__ == '__main__':
    check_file()
