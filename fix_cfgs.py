import glob 
import json 
import os 
files = glob.glob("cfgs/fig_cv_performance/*.json")

for file in files:
    with open(file, "r") as f:
        cfg = json.load(f)
    
    cfg["output_path"] = cfg["output_path"].replace("../results", "../figures")
    for model in cfg["models"]:
        if model["name"] == "Null":
            model["star_color"] = "grey"
            model["cm_color"] = "grey"
    
    with open("../fig_cv_performance/%s" % os.path.basename(file), "w") as f:
        json.dump(cfg, f, indent=4)