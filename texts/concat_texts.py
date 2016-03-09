import os

path = 'AD_TD_full_4letters/'

filenames = sorted([os.path.join(path, fn) for fn in os.listdir(path)])

#with open('4letters_full_corpus.txt', 'w') as outfile:
#    for fname in filenames:
#        with open(fname) as infile:
#            outfile.write(infile.read())
#            outfile.write('\n')

# need to split corpus.txt file with phrases into 80 diff files


#for fname in filenames:
#    with open(fname + '.txt', 'w') as outfile:

fname = filenames[0]
a = 1
with open('full_4letter_phrase_corpus.txt') as infile:
    for c in infile.read():
        with open(str(a) + '.txt', 'w') as outfile:
            #outfile.write(c)
            if c == '\n':
                print a, c
                a += 1
