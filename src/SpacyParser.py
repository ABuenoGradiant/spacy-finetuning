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

                span = doc.char_span(start, end, label=label, alignment_mode="expand")
    
                if span is None:
                    # print(doc.text[start:end])
                    # print(text[start:end])
                    # span = doc.char_span(start, end, label=label, alignment_mode="expand")
                    # print(span.text)
                    print("Skipping entity",text[start:end], label)
                    pass
                else:
                    if any([ent for ent in ents if (start>ent.start_char and start<ent.end_char) or (end>ent.start_char and end<ent.end_char)]):
                    
                        included_entities=[ent for ent in ents if (start>ent.start_char and start<ent.end_char) or (end>ent.start_char and end<ent.end_char)]+[span]
                        longer=sorted(included_entities,key=lambda x: len(x.text))[-1]
                        # print(included_entities,longer)    
                        included_entities.remove(span)                   
                        for included_entity in included_entities:
                            # print("+",start,end,label,text[start:end])
                            # print("-",included_entity.start_char,included_entity.end_char,included_entity.label_,span)
                            ents.remove(included_entity)
                        ents.append(longer)
                    else:
                        ents.append(span)
                        
            doc.ents = ents # label the text with the ents
            db.add(doc)
        # self.compare(json_out,db)
        return db
    
    def compare(self,original_json,db):
        pass
        # def get_entry_by_text(text,db_json):
            
        # db_json=self.to_json(db)
        # original_texts=[line[0] for line in original_json]
        # print("lost text",len([entry for entry in db_json if entry["text"] not in original_texts]))
        # for text,annot in original_json:
            
        
    def to_json(self,doc_bin:DocBin):
        examples = []  # examples in Prodigy's format
        for doc in doc_bin.get_docs(self.nlp.vocab):
            spans = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]
            examples.append({"text": doc.text, "spans": spans})
        return examples
