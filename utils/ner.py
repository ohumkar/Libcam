import nltk
import pandas as pd
import spacy



print("Imported successfully!")

text = "Advanced engineering mathematics in Google located in USA is version abriged edition by Omkar Bhalerao"

def ner_nltk(text, binary = False):
    words = nltk.word_tokenize(text)
    print('BREAK INTO WORDS:\n',words)

    pos_tags  = nltk.pos_tag(words)
    print('\n\n POS TAGS:\n',pos_tags)

    #nltk.help.upenn_tagsest('\n\nNNP')
    print('\n\n CHUNKS:\n')
    chunks = nltk.ne_chunk(pos_tags, binary = False)
    for chunk in chunks :
        print(chunk)

    entities = []
    labels = []
    for chunk in chunks :
        if hasattr(chunk, 'label') :
            entities.append(' '.join(c[0] for c in chunk))
            labels.append(chunk.label())
    entities_labels = list(set(zip(entities, labels)))
    entities_df = pd.DataFrame(entities_labels)
    try :
        entities_df.columns = ['Entities', 'Labels']
        print(entities_df)
    except ValueError as error :
        print(error)
        print('No named found! Try Capitalizing proper nouns')

def ner_spacy(text) :
    print(spacy.__version__)
    assert spacy.util.is_package("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    entities = []
    labels = []
    position_start = []
    position_end = []

    for ent in doc.ents :
        entities.append(ent)
        labels.append(ent.label_)
        position_start.append(ent.start_char)
        position_end.append(ent.end_char)

    df = pd.DataFrame({'Entities': entities, 'Labels': labels, 
                        'Position_Start' : position_start, 'Position_End': position_end})
    print(df)
    spacy.explain('PERSON')
    return df


if __name__ == '__main__' :
    ner_spacy(text)