import json
import os 

paths = ['nbs/tutorials/',
         'nbs/']

for path in paths:

    files = [i for i in os.listdir(path) if i[-5:]=='ipynb']

    for file in files:
        j = json.load(open(f'{path}{file}'))
        j['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}

        with open(f'{path}{file}', 'w') as f:
            json.dump(j, f)