with open("whatsapp_kei.txt", encoding="utf-8") as orig_file:
    file = orig_file.read()
    orig_file.close()

file = file.split('\n')
for row in file:
    if row == '':
        file.remove(row)
    else:
        pass
print(file)
for line in file:
    print(line)
    # if '[' not in line:
    #     print(line)
