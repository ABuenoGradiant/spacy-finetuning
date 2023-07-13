import mlflow
import argparse
import os
import configparser
from datetime import datetime
from utils import CURRENT_PATH,config_file



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
    eval_path=os.path.join(out_path,"evaluation.txt")
    out={}
    with open(eval_path,"r") as eval_file:
        for line in eval_file.read().split("\n"):
            items=[item for item in line.split(" ") if item!=""]
            if len(items)==3:
                if items[0]=="NER":
                    out["{}.{}".format(items[0],items[1])]=float(items[2])

            elif len(items)==2:
                if items[0]=="SPEED":
                    out["WPS"]=float(items[1])

            elif len(items)==4:
                out["{}.{}".format(items[0],"P")]=float(items[1])
                out["{}.{}".format(items[0],"R")]=float(items[2])
                out["{}.{}".format(items[0],"F")]=float(items[3])
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

if __name__=="__main__":
    CONFIG=config_file()
    try:
        args = cmdline_args()
        if args.training_name in CONFIG:
            training_name=args.training_name
            training_config=CONFIG[training_name]
            data_path=os.path.join(CURRENT_PATH,training_config["data_path"]) if not training_config["data_path"].startswith(os.sep) else os.path.join(training_config["data_path"],training_name) 
            out_path=os.path.join(CURRENT_PATH,training_config["out_path"],training_name) if not training_config["out_path"].startswith(os.sep) else os.path.join(training_config["out_path"],training_name) 
            spacy_config_path=os.path.join(CURRENT_PATH,training_config["spacy_config"]) if not training_config["spacy_config"].startswith(os.sep) else os.path.join(training_config["spacy_config"]) 
            ml_experiment=training_config["ml_experiment"]
            
            metrics=read_evaluation(out_path)
            params=read_config(spacy_config_path)
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


