import os

path = 'AD_TD_full_4letters/'

filenames = sorted([os.path.join(path, fn) for fn in os.listdir(path)])

with open('4letters_full_corpus.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            outfile.write(infile.read())
            outfile.write('\n')