try:
    with open('tensorus/tensor_storage.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f'Total lines: {len(lines)}')
        for i in range(508, 512):
            if i < len(lines):
                print(f'Line {i+1}: {repr(lines[i])}')
except Exception as e:
    print(f'Error reading file: {e}')
