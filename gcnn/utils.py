import pickle

def read_data(train_file, test_file, min_frequency, ignore_sentence_length, min_sentence_length):
    tmp_label = []
    tmp_train = []
    frequency = {}
    for line in open(train_file, 'r',encoding="utf-8"):
        data = line.strip().split('\t')
        if len(data) < 2:
            continue
        label = data[0]
        body = data[1].split()
        if len(body) < ignore_sentence_length:
            continue
        tmp_label.append(int(label))
        tmp_train.append(body)
        for term in body:
            if term in frequency:
                frequency[term] += 1
            else:
                frequency[term] = 1

    train_label = [[1] if x > 0 else [0] for x in tmp_label]

    word2id = {}
    id = 2
    for key, value in frequency.items():
        if value > min_frequency - 1:
            word2id[key] = id
            id += 1

    train_data = []
    for text in tmp_train:
        train_data.append([[[word2id[x] if x in word2id else 1 for x in text]]])
        for i in range(min_sentence_length - len(train_data[-1][-1][-1])):
            train_data[-1][-1][-1].append(0)
        
    tmp_label = []
    tmp_test = []
    for line in open(test_file, 'r',encoding="utf-8"):
        data = line.strip().split('\t')
        if len(data) < 2:
            continue
        body = data[1].split()
        if len(body) < min_sentence_length:
            continue
        tmp_label.append(int(data[0]))
        tmp_test.append(body)

    test_label = [1 if x > 0 else 0 for x in tmp_label]
    test_data = []
    for text in tmp_test:
        test_data.append([[[word2id[x] if x in word2id else 1 for x in text]]])
        for i in range(min_sentence_length - len(test_data[-1][-1][-1])):
            test_data[-1][-1][-1].append(0)
        
    with open('word2id.dump', 'wb') as f:
        pickle.dump(word2id, f)

    return id, train_data, train_label, test_data, test_label

def read_embed_matrix(f):
    embed_matrix = []
    for line in f:
        embed_matrix.append([float(x) for x in line.strip().split()])
    return(embed_matrix)

