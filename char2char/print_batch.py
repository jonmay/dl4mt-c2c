import pprint
import sys
import numpy as np

def pbatch(source, dic, logfile=sys.stdout):
    ss = np.transpose(source)
    for line in ss[:10]:
        for word in line:
            a = dic[word]
            b = a

            if a == "SOS":
                b = "{"
            elif a == "EOS":
                b = "}"
            elif a == "ZERO":
                b = "_"
            elif a == "UNK":
                b = "|"

            logfile.write(b)
        print >>logfile," "
    print >>logfile,""

def pbatch_many(source, dic, n_x, logfile=sys.stdout):
    ss = np.transpose(source)
    iis = [0, 20, n_x-8,n_x-1]

    for ii in iis:
        line = ss[ii]
        for word in line:
            a = dic[word]
            b = a

            if a == "SOS":
                b = "{"
            elif a == "EOS":
                b = "}"
            elif a == "ZERO":
                b = "_"
            elif a == "UNK":
                b = "|"

            logfile.write(b)
        print >>logfile," "
    print >>logfile,""
