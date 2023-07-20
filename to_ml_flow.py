import mlflow
import argparse
import os
import configparser
from datetime import datetime
from utils import CURRENT_PATH,config_file,read_jsonl,read_json,save_json,save_jsonl
import spacy
from spacy.scorer import Scorer
from spacy.training import Example
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import json

def cmdline_args():
        # Make parser object
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    p.add_argument("training_name",
                   help="Traning name in config.")
     
    # p.add_argument("required_int", type=int,
    #                help="req number")
    # p.add_argument("--on", action="store_true",
    #                help="include to enable")
    # p.add_argument("-v", "--verbosity", type=int, choices=[0,1,2], default=0,
    #                help="increase output verbosity (default: %(default)s)")
                   
    # group1 = p.add_mutually_exclusive_group(required=True)
    # group1.add_argument('--enable',action="store_true")
    # group1.add_argument('--disable',action="store_false")

    return(p.parse_args())


def to_ml_flow(params,metrics,experiment,run_name):
    #     # Establecer la URL del servidor MLflow
    mlflow.set_tracking_uri("http://51.178.73.104:5000/")
    
    existing_exp = mlflow.get_experiment_by_name(experiment)
    if not existing_exp:
        mlflow.create_experiment(experiment)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("run_name",run_name)
        mlflow.end_run()
    pass

def read_evaluation(out_path):
    eval_path=os.path.join(out_path,"evaluation.json")
    with open(eval_path,"r") as json_in:
        data=json.load(json_in)

    out={}
    for key,items in data["ents_per_type"].items():
        for key2,value in items.items():
            out[key+"."+key2]=value
    return out

def read_config(config_path):
    config_spacy = configparser.ConfigParser()
    config_spacy.read(config_path)
    return {
        "lang": config_spacy["nlp"]["lang"],
        "batch_size": config_spacy["nlp"]["batch_size"],
        "dropout": config_spacy["training"]["dropout"],
        "learn_rate": config_spacy["training.optimizer"]["learn_rate"], 
        "model": config_spacy["components.ner"]["source"]          
        }

def get_predictions(ner_model, examples):
    
    def get_tag(start,end,entities):
        for ent in entities:
            ent_start=ent[0]
            ent_end=ent[1]
            
            if start>=ent_start and start<ent_end:
                return "I-"+ent[2]
            elif start<ent_start and end>ent_end:
                return "I-"+ent[2]
        return "O"
            

    y_preds=[]
    y_trues=[]
    labels=[]
    
    for input_, annot in tqdm(examples,desc="Predicting"):
        pred_value = ner_model(input_.replace("\n"," "))
        y_pred=[(w.text,str(w.ent_iob_.replace("B","I")+"-"+w.ent_type_).strip("-")) for w in pred_value]
        y_true=[(w.text,get_tag(w.idx,w.idx+len(w.text),annot)) for w in pred_value]
        # print("TRUE",[(w.text,w.idx,w.idx+len(w.text),get_tag(w.idx,w.idx+len(w.text),annot)) for w in pred_value])
        # print("PRED",[(w.text,str(w.ent_iob_.replace("B","I")+"-"+w.ent_type_).strip("-")) for w in pred_value])
        # print(annot)
        y_preds.append(y_pred)
        y_trues.append(y_true)
        labels.extend(list(set([label[1] for  label in y_pred + y_true])))
        labels=list(set(labels))
    
    
    return y_preds,y_trues,labels

def get_scorer(ner_model, test_data):
    scorer=Scorer()
    examples=[]
    for text, annotation in tqdm(test_data,desc="Scoring..."):
        doc_pred=ner_model(text)

        example = Example.from_dict(doc_pred, {"entities":annotation})
        examples.append(example)
        
    scores = scorer.score_spans(examples, "ents")
    return scores

def generate_confusion_matrix(training_name,config,exclude_tags=["O"]):

    model_path=os.path.join(CURRENT_PATH,config["out_path"],training_name,"model-best")
    spacy_model=spacy.load(model_path)
    
    test_path=os.path.join(CURRENT_PATH,config["data_path"],"test.json")
    test=read_jsonl(test_path)
    
    scores=get_scorer(spacy_model,test)
    save_json(os.path.join(CURRENT_PATH,config["out_path"],training_name,"original_scores.json"),scores)
        
    y_preds,y_trues,labels=get_predictions(spacy_model,test)
    with open(os.path.join(CURRENT_PATH,config["out_path"],training_name,"predictions.txt"),"w+") as out_prediction:
        predictions_txt=""
        for y_true,y_pred in zip(y_trues,y_preds):
            predictions_txt+="\n"
            for t_true, t_pred in zip(y_true,y_pred):
                predictions_txt+="{}\t{}\t{}\n".format(t_true[0],t_pred[1],t_true[1])
        out_prediction.write(predictions_txt.strip("\n"))
    
    y_true=[et[1] for t,p in zip(y_trues,y_preds) for et,ep in zip(t,p) if not (et[1]==ep[1] and et[1] in exclude_tags)]
    y_pred=[ep[1] for t,p in zip(y_trues,y_preds) for et,ep in zip(t,p) if not (et[1]==ep[1] and et[1] in exclude_tags)]
    
    cm=confusion_matrix(y_true,y_pred,labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()
    plt.savefig(os.path.join(CURRENT_PATH,config["out_path"],training_name,"cm.png"))

if __name__=="__main__":
    CONFIG=config_file()
    try:
        args = cmdline_args()
        if args.training_name in CONFIG:
            
            #config paths
            training_name=args.training_name
            training_config=CONFIG[training_name]
            data_path=os.path.join(CURRENT_PATH,training_config["data_path"]) if not training_config["data_path"].startswith(os.sep) else os.path.join(training_config["data_path"],training_name) 
            out_path=os.path.join(CURRENT_PATH,training_config["out_path"],training_name) if not training_config["out_path"].startswith(os.sep) else os.path.join(training_config["out_path"],training_name) 
            spacy_config_path=os.path.join(CURRENT_PATH,training_config["spacy_config"]) if not training_config["spacy_config"].startswith(os.sep) else os.path.join(training_config["spacy_config"]) 
            ml_experiment=training_config["ml_experiment"]
            
            #confusion matrix
            generate_confusion_matrix(training_name,training_config)
            
            #configure data to mlflow
            metrics=read_evaluation(out_path)
            training_stats=read_json(os.path.join(out_path,"training_stats.json"))
            
            
            params=read_config(spacy_config_path)
            params=dict(params,**training_stats)
            
            now=datetime.now().strftime("%d%m%Y-%H%M%S")
            run_name="{}_{}".format(training_name,now)
            print("Saving",metrics,params,run_name)
            to_ml_flow(params,metrics,ml_experiment,run_name)
            print("Done! save in mlflow.")
        else: 
            print("No {} in config file. Try: {}".format(args.training_name,",".join(list(CONFIG.keys()))))
        
    except Exception as e:
        print('TRY: python3 to_ml_flow.py example_project')
        raise(e)


