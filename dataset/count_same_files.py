import os

script_path = os.path.dirname(os.path.realpath(__file__)) 
path = script_path + '/files/'

files = os.listdir(path)

raw_files = []

for file in files :
	with open(path+file) as f :
		raw_files.append(f.read())


with open(script_path + '/same_files.txt', 'w') as f :
	for i in range(0, len(raw_files)) :
		for j in range(i+1, len(raw_files)) :
			if raw_files[i] == raw_files[j] :
				f.write(f'{files[i]} == {files[j]}\n')
