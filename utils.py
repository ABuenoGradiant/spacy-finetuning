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

def read_json(file_path:str):
    with open(file_path,"r") as json_in:
        return json.load(json_in)
    
def save_jsonl(file_path:str,data:list):
    with open(file_path,"w+") as out_json:
        for line in data:
            json.dump(line,out_json,ensure_ascii=False)
            out_json.write("\n")

def save_json(file_path:str,data:dict):
    with open(file_path,"w+") as out_json:
        json.dump(data,out_json,ensure_ascii=False)


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
   