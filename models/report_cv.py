import numpy as np 
import os 
import sys 
import utils.eval_funcs as eval_funcs 

np.set_printoptions(precision=3)

def main():
    
    cv_dir = sys.argv[1]
    
    r = eval_funcs.average_results(cv_dir)
    
    eval_funcs.print_eval_classifier(r)

    
if __name__ == "__main__":
    main()
