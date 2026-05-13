import sys
path = sys.argv[1]
with open(path, 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'ctx.language = customer.language_pref' in line and 'user_language' not in line:
        indent = line[:len(line) - len(line.lstrip())]
        lines.insert(i + 1, indent + 'ctx.user_language = customer.language_pref or "en"\n')
        break
with open(path, 'w') as f:
    f.writelines(lines)
print("Done - inserted user_language line")
