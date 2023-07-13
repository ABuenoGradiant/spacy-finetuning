# argsparse from https://gist.github.com/ahogen/6fc1760bbf924f4ee6857a08e4fea80a
import sys,os
import argparse
from utils import config_file,CURRENT_PATH,read_jsonl

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










if __name__ == '__main__':

    try:
        args = cmdline_args()
        if args.training_name in CONFIG:
            training_name=args.training_name
            training_config=CONFIG[training_name]
            data_path=os.path.join(CURRENT_PATH,training_config["data_path"]) if not training_config["data_path"].startswith(os.sep) else os.path.join(training_config["data_path"],training_name) 
            out_path=os.path.join(CURRENT_PATH,training_config["out_path"],training_name) if not training_config["out_path"].startswith(os.sep) else os.path.join(training_config["out_path"],training_name) 
            spacy_config_path=os.path.join(CURRENT_PATH,training_config["spacy_config"]) if not training_config["spacy_config"].startswith(os.sep) else os.path.join(training_config["spacy_config"]) 
            
            os.makedirs(out_path,exist_ok=True)
            spacy_parser=SpacyParser(training_config["model"])
            
            paths={}
            for file in os.listdir(data_path):
                if file in ["train.json","dev.json","test.json"]:
                    name=file.replace(".json","")
                    json_out=read_jsonl(os.path.join(data_path,file))
                    spacy_db=spacy_parser.to_spacy_format(json_out)
                    spacy_file="{}.spacy".format(name)
                    spacy_db.to_disk(os.path.join(out_path,"{}.spacy".format(name))) 
                    paths[name]=os.path.join(out_path,"{}.spacy".format(name))

            with open(os.path.join(out_path,"run_train.sh"),"w+") as bash_out:
                
                bash_out.write(
                "python3 -m spacy train {} --output {} --paths.train {} --paths.dev {}".format(
                    spacy_config_path, out_path, paths["train"],paths["dev"]
                ))
                bash_out.write("\n")
                bash_out.write(
                "python3 -m spacy evaluate {} {} > {}".format(
                    os.path.join(out_path,"model-best"), paths["test"],os.path.join(out_path,"evaluation.txt")
                ))
                bash_out.write("\n")
                bash_out.write("python3 {} {}".format(os.path.join(CURRENT_PATH,"to_ml_flow.py"),training_name))
            
            print("Done! please run: \nbash {}\n to train the model.".format(os.path.join(out_path,"run_train.sh")))
        else: 
            print("No {} in config file. Try: {}".format(args.training_name,",".join(list(CONFIG.keys()))))
        
    except Exception as e:
        
        print('TRY: python3 generate_spacy_training.py example_project')
        raise(e)

    print()