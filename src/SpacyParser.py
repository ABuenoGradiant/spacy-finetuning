from spacy.tokens import DocBin
from tqdm import tqdm
import spacy,random
#nlp = spacy.blank("en") # load a new spacy model


class SpacyParser():
    def __init__(self,model) -> None:
        self.nlp = spacy.load(model) # load other spacy model


    def to_spacy_format(self,json_out:list):

        db = DocBin() # create a DocBin object
        
        for entry in tqdm(json_out): # data in previous format
            text=entry[0]
            annot=entry[1]
            doc = self.nlp.make_doc(text) # create doc object from text
            ents = []
            for start, end, label in annot: # add character indexes
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    # print("Skipping entity",label)
                    pass
                else:
                    ents.append(span)
            doc.ents = ents # label the text with the ents
            db.add(doc)
        return db
