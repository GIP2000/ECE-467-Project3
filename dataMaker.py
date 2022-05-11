from random import shuffle

with open("./realdonaldtrump.csv", 'r') as f, open("./data3.txt", 'w') as out:
    first = True
    data = ""
    for line in f:
        if first:
            first = False
            continue
        field = line.split(',')
        data += field[2] + " <TE> "
    data = "".join(data.split('"'))
    data = " ".join([l for l in data.split(" ") if "http" not in l])
    corpus = data.split("<TE>")
    corpus = corpus[0:len(corpus)//5]
    shuffle(corpus)
    out.write("<TE>".join(corpus))

