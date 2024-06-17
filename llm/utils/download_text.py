import requests
import os
import sys

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
WDIR = os.path.dirname(os.path.abspath(__file__))


response = requests.get(url)
if response.status_code == 200:

	text_content = response.text
	file_name = sys.argv[1]
	file_path = os.path.join(WDIR, "../data", file_name + ".txt")

	with open(file_path, 'w', encoding='utf-8') as file:
		file.write(text_content)

	print(f"Text successfully written to {file_path}")
