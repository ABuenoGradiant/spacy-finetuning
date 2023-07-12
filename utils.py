import json,os,yaml

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

def config_file():
    
    config_vars = read_yaml(
        os.path.join(CURRENT_PATH, "config.yml")
    )
    return config_vars

def read_yaml(_path):

    with open(_path, "r") as f_stream:
        config = yaml.load(f_stream, Loader=yaml.FullLoader)

    return config

def read_jsonl(file_path:str):
    
    with open(file_path,"r",encoding="utf8") as json_in:
        out=[]
        
        for line in json_in:
            try:
                entry=json.loads(line)
                out.append(entry)
            except:
                continue
        
        return  out

def get_info_entities(data:list):
    out={}
    for entry in data:
        tags=[t[2] for t in entry[1]]
        for t in tags:
            if t not in out:
                out[t]=0
            out[t]+=1
    return out

def get_info_lines(data:list):
    out={}
    for entry in data:
        num_entities=len(entry[1])
        if num_entities not in out:
            out[num_entities]=0
        out[num_entities]+=1
    return out
   