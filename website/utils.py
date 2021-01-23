import numpy as np 
import json 

def pack_sgo(terms):
    """ packs sGO terms into array of """
    packed = np.packbits(terms).tobytes()
    return packed 

def unpack_sgo(buf, n_terms):
    """ unpacks sGO terms back to numpy bool array """
    unpacked = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), count=n_terms)
    return unpacked.astype(bool)


if __name__ == "__main__":

    read_gene_features('cfgs/models/yeast_gi_mn.json')

