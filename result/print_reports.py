import json
import os

def print_report(fname, category = 'Classification Report'):
    with open(fname + '.json') as f:
        dic = json.load(f)  

    print(dic[category])

print(os.listdir('.'))

print_report(input())