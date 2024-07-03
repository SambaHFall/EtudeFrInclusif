import os

script_path = os.path.dirname(os.path.realpath(__file__)) 
sur_script_path = os.path.dirname(script_path)
path = sur_script_path + '/dataset/files/'

"""
Produce a same_files.txt file, which enumerate all the group of files in etude_fr_inclusif/dataset/files that are exactly the same
"""

files = os.listdir(path)

raw_files = []

for file in files :
	with open(path+file) as f :
		raw_files.append(f.read().strip())


files_dict = {}

with open(sur_script_path + '/same_files.txt', 'w') as f :
	for i in range(0, len(raw_files)) :
		for j in range(i+1, len(raw_files)) :
			if raw_files[i] == raw_files[j] :
				if i not in files_dict :
					files_dict[i] = [files[i], files[j]]
					files_dict[j] = i
				else :
					root = files_dict[i]
					if isinstance(root, list) :
						files_dict[j] = i
						files_dict[i].append(files[j])
					else :
						files_dict[j] = root
						files_dict[root].append(files[j])


with open(script_path + '/same_files.txt', 'w') as f :
	for value in files_dict.values() :
		if isinstance(value, list):
			f.write(str( list(set(value)) ) + '\n')
