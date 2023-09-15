# argsparse from https://gist.github.com/ahogen/6fc1760bbf924f4ee6857a08e4fea80a
import sys,os,json
import argparse
from tqdm import tqdm

from utils import config_file,CURRENT_PATH,read_jsonl,save_jsonl,save_json
from src.SpacyParser import SpacyParser

CONFIG=config_file()


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




def check_spans(data:list):
    for entry in data:
        labels=entry[1]
        
        for label in labels:
            nested_labels=[ l for l in labels if label!=l and l[0]>=label[0] and l[1]<=label[1]]
            if any(nested_labels):

                if len(nested_labels)==1:
                    nested_label=nested_labels[0]
                    if  nested_label[1]-nested_label[0]==1:
                        if nested_label in entry[1]:
                            entry[1].remove(nested_label)
                    else:    
                        print(nested_labels,entry[0][nested_label[0]:nested_label[1]])
                        if nested_label[2]=="MISC" and label[2]!="MISC":
                            entry[1].remove(nested_label)
                            print(nested_label, "removed!")
                else:
                    print(">",label,entry[0][label[0]:label[1]])
                    print(nested_labels)
                
                
    return data





if __name__ == '__main__':

    try:
        args = cmdline_args()
        if args.training_name in CONFIG:
            
            #reading paths
            training_name=args.training_name
            training_config=CONFIG[training_name]
            data_path=os.path.join(CURRENT_PATH,training_config["data_path"]) if not training_config["data_path"].startswith(os.sep) else os.path.join(training_config["data_path"],training_name) 
            out_path=os.path.join(CURRENT_PATH,training_config["out_path"],training_name) if not training_config["out_path"].startswith(os.sep) else os.path.join(training_config["out_path"],training_name) 
            spacy_config_path=os.path.join(CURRENT_PATH,training_config["spacy_config"]) if not training_config["spacy_config"].startswith(os.sep) else os.path.join(training_config["spacy_config"]) 
            spacy_bins_path=os.path.join(out_path,"spacy_bins")
            spacy_files_path=os.path.join(out_path,"spacy_files")
            os.makedirs(spacy_bins_path,exist_ok=True)
            os.makedirs(spacy_files_path,exist_ok=True)
            
            #loading  ner
            spacy_parser=SpacyParser(training_config["model"])
            
            paths={}
            training_stats={}
            for file in tqdm(os.listdir(data_path)):
                if file.startswith(("train","dev","test")):
                    
                    name=file.replace(".json","")
                    spacy_file="{}.spacy".format(name)

                    json_out=read_jsonl(os.path.join(data_path,file))
                    # json_out=check_spans(json_out)
                    # generate spacy bin files
                    spacy_db=spacy_parser.to_spacy_format(json_out)
                    spacy_db.to_disk(os.path.join(spacy_bins_path,spacy_file)) 
                    
                    out_spacy=spacy_parser.to_json(spacy_db)
                    save_jsonl(os.path.join(spacy_files_path,file),out_spacy)
                    paths[name]=os.path.join(spacy_bins_path,"{}.spacy".format(name))
                    
                    #training stats process
                    training_stats[file.split(".")[0]]={}
                    original_labels=[ent[2] for entry in json_out for ent in entry[1]]
                    labels=list(set(original_labels))
                    original_labels_count={label:original_labels.count(label) for label in labels}
                    training_stats[file.split(".")[0]]["original"]=dict(original_labels_count,**{"len":len(json_out)})
                    
                    spacy_labels=[ent["label"] for entry in out_spacy for ent in entry["spans"]]
                    labels=list(set(spacy_labels))
                    spacy_labels_count={label:spacy_labels.count(label) for label in labels}
                    training_stats[file.split(".")[0]]["spacy"]=dict(spacy_labels_count,**{"len":len(out_spacy)})
                    
            save_json(os.path.join(out_path,"training_stats.json"),training_stats)
            
            ## write execution bash
            with open(os.path.join(out_path,"run_train.sh"),"w+") as bash_out:
                
                bash_out.write(
                "python3 -m spacy train {} --output {} --paths.train {} --paths.dev {}".format(
                    spacy_config_path, out_path, paths["train"],paths["dev"]
                ))
                bash_out.write("\n")
                bash_out.write(
                "python3 -m spacy benchmark accuracy {} {} -o {}> {}".format(
                    os.path.join(out_path,"model-best"), paths["test"],os.path.join(out_path,"evaluation.json"),os.path.join(out_path,"evaluation.txt")
                ))
                bash_out.write("\n")
                
                for test_name in paths.keys():
                    if test_name.startswith("test_"):
                        lang=test_name.split("_")[-1]
                        bash_out.write(
                            "python3 -m spacy benchmark accuracy {} {} -o {}".format(
                            os.path.join(out_path,"model-best"), paths[test_name],os.path.join(out_path,"evaluation_{}.json".format(lang))
                            ))
                        bash_out.write("\n")
                
                bash_out.write("python3 {} {}".format(os.path.join(CURRENT_PATH,"to_ml_flow.py"),training_name))
            
            print("Done! please run: \nbash {}\n to train the model.".format(os.path.join(out_path,"run_train.sh")))
        else: 
            print("No {} in config file. Try: {}".format(args.training_name,",".join(list(CONFIG.keys()))))
        
    except Exception as e:
        
        print('TRY: python3 generate_spacy_training.py example_project')
        raise(e)