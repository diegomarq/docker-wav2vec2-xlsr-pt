#!/usr/bin/python3

import subprocess, os, sys
from shutil import copyfile

def convert_to(file, output_path, ext):
	output_file = os.path.join(output_path, os.path.splitext(os.path.basename(file))[0] + '.' + ext)
	
	process = subprocess.Popen([\
		'ffmpeg',\
		'-y',\
		'-i',\
		file,\
		'-ar', \
		'16000',\
		'-ac',\
		'1',\
		'-f',\
		ext,\
		output_file],\
		stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

	return output_file

if __name__ == "__main__":
	input_path = sys.argv[1]
	output_path = sys.argv[2]
	ext = sys.argv[3]

	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	print(f"Looking path {input_path}")

	if os.path.isdir(input_path):
		for name in os.listdir(input_path):
			file = os.path.join(input_path, name)

			if (os.path.splitext(file)[-1] == '.' + ext):
				print(f"File with ext {file}")
				copyfile(file, os.path.join(output_path, name))
			else:
				print(f"Converting {file}")
				output_file = convert_to(file, output_path, ext)

				if os.path.isfile(output_file):
					print("OK")

	print("Finish!")