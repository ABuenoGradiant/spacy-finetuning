# spacy-finetuning
Module to finetuning a spacy model. 

# Confing a finetuning entry
In the file config.yml you 
# Models to finetuning
It is mandatory to install the spacy models you want to install via requirements.txt. For example:
```
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl
es-core-news-lg @ https://github.com/explosion/spacy-models/releases/download/es_core_news_lg-3.5.0/es_core_news_lg-3.5.0-py3-none-any.whl
```
This two model are already added. If you want to use other, please check:
https://github.com/explosion/spacy-models

or run:
```
python3 -m spacy download en_core_web_sm
```

# Data 

The data format is a common data for NER training. The training folder should contains 3 files: train.json, dev.json and test.json. Each file should be a JSONL file with one list per line: the first element mush be the text line and the second element mush be another list with its entities in the format [tag,init,end]. Here is an example:
```
["El poeta, narrador i traductor Josep Piera, 55è Premi d'Honor de les Lletres Catalanes\n- Racó Català Inici  Notícies  Cultura, Llengua Notícia Activista per la llengua d'una dilatada trajectòria literària, ha fet de La Safor un dels motius de la seva obra El poeta, narrador, assagista, articulista i traductor Josep Piera Autor/a: Pere Francesch El poeta, narrador, assagista, articulista i traductor Josep Piera (Beniopa, La Safor, 1947) ha estat distingit amb el 55è Premi d'Honor de les Lletres Catalanes que atorga Òmnium Cultural.", [[31, 42, "PER"], [48, 55, "MISC"], [56, 86, "MISC"], [89, 100, "ORG"], [218, 226, "LOC"], [313, 324, "PER"], [334, 348, "PER"], [404, 415, "PER"], [417, 424, "LOC"], [426, 434, "LOC"], [472, 510, "MISC"]]]
```

You can check an example in data/example

# Spacy config

Spacy has a tool to generate the configuration file for training. You can check it here: 
https://spacy.io/usage/training
However, you can find some examples in the formder spacy_config.



