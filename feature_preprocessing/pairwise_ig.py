import numpy as np 
import igraph as ig
import sys 
import os 
import multiprocessing 
import numpy.random as rng 

thismodule = sys.modules[__name__]

def main(gpath, feature_func, n_cpu=6):
    
    ff = getattr(thismodule, feature_func)

    G = ig.read(gpath)
    G_labels = sorted([G.vs['label'][i] for i in range(G.vcount())])
    node_ix = dict(zip(G_labels, range(G.vcount())))

    fargs = [(G, ff, i) for i in range(G.vcount())]
    with multiprocessing.Pool(n_cpu) as pool:
        results = pool.map(calculate, fargs)
        results = np.array(results)
        print(results.shape)

    # fix indexing
    ix_pairs = [(node_ix[G.vs['label'][i]], i) for i in range(G.vcount())]
    node_ix_to_vid = dict(ix_pairs)
    fixed_ix = [node_ix_to_vid[i] for i in range(G.vcount())]
    F = results[fixed_ix, :]
    F = F[:, fixed_ix]

    # make it symmetric
    F = F + F.T 
    
    # checking
    # N = 100
    # for i in range(N):
    #     src_nid = rng.choice(G.vcount())
        
    #     for tar_nid in range(0, G.vcount()):
    #         if tar_nid == src_nid:
    #             continue
    #         src_vid = node_ix_to_vid[src_nid]
    #         tar_vid = node_ix_to_vid[tar_nid]
    #         val = ff(G, src_vid, tar_vid)
    #         if val > 0:
    #             print("%d %d %d" % (src_nid, tar_nid, val))
    #         assert(val == F[src_nid, tar_nid])
    #         assert(val == F[tar_nid, src_nid])

    #print(check_symmetric(F))
    #print(np.sum(F.diagonal()) == 0)

    output_path = "../generated-data/pairwise_features/%s_%s" % (os.path.basename(gpath).replace('.gml',''), ff.__name__)
    np.save(output_path, F)

# def check_symmetric(a, rtol=1e-05, atol=1e-08):
#     return np.allclose(a, a.T, rtol=rtol, atol=atol)

def calculate(args):
    G, ff, src_vid = args 
    F = np.zeros(G.vcount())
    for tar_vid in range(src_vid+1, G.vcount()):
        F[tar_vid] = ff(G, src_vid, tar_vid)
    print("Done %d" % src_vid)
    return F 

def adhesion(G,a,b):
    return G.adhesion(a, b)

def cohesion(G,a,b):
    return G.cohesion(a, b, neighbors="infinity")

def adjacent(G,a,b):
    return G.get_eid(a, b, error=False) > -1

def mutual_neighbors(G,a,b):
    n1 = G.neighbors(a)
    n2 = G.neighbors(b)
    return len(set(n1).intersection(n2))

if __name__ == "__main__":
    gpath = sys.argv[1]
    feature_func = sys.argv[2]
    

    main(gpath, feature_func)
