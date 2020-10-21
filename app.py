from flask import Flask, request
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from os import path, listdir, makedirs

CORPUS = "./volume/corpus"
DATA = path.join(CORPUS, "data")

if not path.isdir(CORPUS) or len(listdir(CORPUS)) == 0:
    makedirs(CORPUS, exist_ok=True)
    data_utils.download_data(CORPUS)

ws = WS(DATA)
pos = POS(DATA)
ner = NER(DATA)
app = Flask(__name__)

@app.route('/tagger', methods=['POST'])
def tagger():
    sentences = request.json["raw"]
    ws_list = ws(sentences)
    pos_list = pos(ws_list)
    entity_list = ner(ws_list, pos_list)
    result = []
    for i in range(len(sentences)):
        words = []
        entities = {}
        for w, p in zip(ws_list[i], pos_list[i]):
            words.append({ "w": w, "p": p })
        for entity in entity_list[i]:
            kind = entity[2]
            val = entity[3]
            if kind in entities:
                entities[kind].append(val)
            else:
                entities[kind] = [val]
        result.append({ "words": words, "entities": entities })
    return { "result": result }
