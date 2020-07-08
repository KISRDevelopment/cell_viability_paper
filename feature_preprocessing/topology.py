import numpy as np
import networkx as nx
import collections
import sys
import os
import igraph as ig
import scipy.stats as stats
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats
thismodule = sys.modules[__name__]

nx_features = [
    'mnc',
    'lac',
    'l_index',
    'subgraph_centrality',
    'degree_centrality',
    'eigenvector_centrality',
    'katz_cent',
    'load_centrality',
    'harmonic_centrality',
    'clustering',
    'information_centrality',
    'lid',
    'ecc'
]
ig_features = [
    'betweenness',
    'closeness',
    'cutpoint',
    'constraint', 
    'coreness',
    'eccentricity',
    'hub_score',
    'neighborhood_size',
    'harary',
    'complexity_index'
]

def main():
    graph_path = sys.argv[1]
    
    G = nx.read_gpickle(graph_path)
    nodes = list(sorted(G.nodes()))
    # np.random.shuffle(nodes)
    # nodes = nodes[:50]
    #G = nx.subgraph(G, nodes)

    nodes_to_features = collections.defaultdict(dict)
    extract_features_networkx(G, nx_features, nodes_to_features)

    iG = ig.read(graph_path + '.gml')
    extract_features_ig(iG, ig_features, nodes_to_features)
    
    # features are ordered according to nodes
    F = []
    feature_labels = []
    for feature in nx_features + ig_features:
        Ff = []
        for n in nodes:
            v = nodes_to_features[n]

            Ff.append(v[feature])
        Ff = np.array(Ff)

        if len(Ff.shape) == 1:
            Ff = Ff[:,None]

        if Ff.shape[1] == 1:
            feature_labels.append(feature)
        else:
            for i in range(Ff.shape[1]):
                feature_labels.append("%s_%d" % (feature, i))
        
        F.append(Ff)
    
    F = np.hstack(F)
    
    mu = np.mean(F, axis=0)
    std = np.std(F, axis=0)

    print(mu)
    print(std)

    # normalize
    F = stats.zscore(F, axis=0)
    

    print(F.shape)
    
    print(np.min(F, axis=0))
    print(np.max(F, axis=0))
    print(np.mean(F, axis=0))
    print(np.std(F, axis=0))

    print(feature_labels)

    output_path = '../generated-data/features/%s_topology' % (os.path.basename(graph_path))
    np.savez(output_path, F=F, feature_labels=feature_labels, mu=mu, std=std)

    
def extract_features_networkx(G, features, nodes_to_features):
    # embed nodes using the specified features
    for feature in features:
        
        print("Computing %s" % feature)

        # get the function dynamically by name
        # try the networkx module first, if no func is available with that 
        # name, try the current module.
        if hasattr(nx, feature):
            func = getattr(nx, feature)
        else:
            func = getattr(thismodule, feature)
        
        # execute the function to get a dictionary of nodes -> feature value
        try:
            nodes_to_value = func(G)
        except Exception as e:
            nodes_to_value = { n : 0. for n in G.nodes() }
            print(e)

        for n in nodes_to_value:
            v = nodes_to_value[n]
            if type(v) != list and (np.isnan(v) or np.isinf(v)):
                nodes_to_value[n] = 0
                
        # merge
        for k, v in nodes_to_value.items():
            nodes_to_features[k][feature] = v

def extract_features_ig(G, features, nodes_to_features):
    
    # embed nodes using the specified features
    for feature in features:
        print("Computing %s" % feature)
        
        func = getattr(thismodule, feature)
        F = func(G)
        
        # merge
        for i in range(G.vcount()):
            k = G.vs['label'][i]
            nodes_to_features[k][feature] = F[i,:]
    
# -------------------------------------------------------------------------------------
# custom centralities that rely on networkx
# -------------------------------------------------------------------------------------
def ecc(G):
    
    triangles = get_triangles(G)

    # compute number of triangles each edge is involved in
    ecc_cent = { tuple(sorted(e)): 0 for e in G.edges() }
    for triangle in triangles:
        e_ab = tuple(sorted((triangle[0], triangle[1])))
        e_bc = tuple(sorted((triangle[1], triangle[2])))
        e_ca = tuple(sorted((triangle[2], triangle[0])))

        edges = [e_ab, e_bc, e_ca]
        for edge in edges:
            ecc_cent[edge] += 1
    
    # normalize by max number of triangles
    degree = nx.degree(G)
    for e in G.edges():
        e = tuple(sorted(e))
        if ecc_cent[e] > 0:
            ecc_cent[e] = ecc_cent[e] / min(degree[e[0]]-1, degree[e[1]] - 1)

    # add up the edge clustering coefficient to get NC for each node
    cent = {}
    for n in G.nodes():
        cent[n] = 0
        for e in G.edges(n):
            e = tuple(sorted(e))
            cent[n] += ecc_cent[e]

    return cent

def get_triangles(G):
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    triangles = []
    for (u, v) in G.edges():
        for w in G.nodes():
            if G.has_edge(u, w) and G.has_edge(w, v):
                triangles.append((u, w, v))

    return triangles

def lid(G):
    cent = {}
    for node in G.nodes():
        sG = G.subgraph(G.neighbors(node))
        e_int = sG.number_of_edges()
        v_int = len([n for n in sG.nodes() if sG.degree(n) > 0])
        if v_int > 0:
            cent[node] = e_int / v_int
        else:
            cent[node] = 0
    return cent

def mnc(G):
    cent = {}
    for node in G.nodes():
        neighbors = G.neighbors(node)
        Gsub = G.subgraph(neighbors)
        giant = max(nx.connected_components(Gsub), key=len)
        cent[node] = len(giant)
    return cent 
    
def lac(G):
    cent = {}
    for node in G.nodes():
        neighbors = G.neighbors(node)
        Gsub = G.subgraph(neighbors)
        Gsub_cent = dict(Gsub.degree())
        cent[node] = np.mean(list(Gsub_cent.values()))
    return cent



def l_index(G):
        
    cent = {}
    
    degree = dict(G.degree())
    max_degree = max(degree.values())
    
    for node in G.nodes():
    
        neighbors = list(G.neighbors(node))
        best_h = 0
        for h in range(max_degree):
            if len(neighbors) == 0:
                break
            neighbors = [n for n in neighbors if degree[n] >= h]
            if len(neighbors) >= h:
                best_h = h
        cent[node] = best_h
    return cent
        
def katz_cent(G):

    A = nx.to_numpy_matrix(G)

    e = np.linalg.eigvals(A)

    maxe = np.max(np.real(e))

    alpha = 1/(2*maxe)

    return nx.katz_centrality(G, alpha=alpha)

# -------------------------------------------------------------------------------------
# centralities that rely on igraph
# -------------------------------------------------------------------------------------
def closeness(G):
    return np.array([G.closeness()]).T 

def betweenness(G):
    return np.array([G.betweenness()]).T 

def constraint(G):
    return np.array([G.constraint()]).T
               
def coreness(G):
    return np.array([G.coreness()]).T

def eccentricity(G):
    return np.array([G.eccentricity()]).T

def hub_score(G):
    return np.array([G.hub_score()]).T

def neighborhood_size(G):
    F = []
    for n in [1,2,3]:
        F.append(G.neighborhood_size(order=n))
    return np.array(F).T

def cutpoint(G):
    
    nodes = range(G.vcount())

    betweenness = G.betweenness()
    degree = G.degree(nodes)
    
    df = pd.DataFrame([{ 'betweeness' : betweenness[k],
                         'degree' : degree[k] } for k in nodes])
    model = smf.ols('betweeness ~ degree',  data=df).fit()
    resid = model.predict(df) - df['betweeness']
    
    return np.array([resid]).T 

def harary(G):
    D = G.shortest_paths_dijkstra()
    F = np.max(D, axis=1)
    F = 1/F 
    return F[:,None]

def complexity_index(G):
    nodes = range(G.vcount())
    D = G.shortest_paths_dijkstra()
    degree = G.degree(nodes)
    sum_shortest_path_lens = np.sum(D, axis=1)
    F = degree / sum_shortest_path_lens
    return F[:,None]

if __name__ == "__main__":
    main()
    