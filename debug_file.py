with open('tensorus/tensor_storage.py', 'rb') as f:
    content = f.read()
    
# Check for any unusual characters around line 510 (approximately byte 15000-15200)
start = 15000
end = 15200
section = content[start:end]
print(f'Bytes {start}-{end}:')
print(repr(section))

# Also check the exact line
lines = content.split(b'\n')
if len(lines) > 510:
    print(f'Line 510 (bytes): {repr(lines[509])}')
    print(f'Line 510 (decoded): {lines[509].decode("utf-8", errors="replace")}')
