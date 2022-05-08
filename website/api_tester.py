import requests 
import json 


def main():

    #test_gi_pairs()
    
    #test_interpret()
    #test_common_interactors()
    test_gi_triplets()
def test_gi_triplets():

    rp = {
        "threshold" : 0.9,
        "gene_a" : "snf1",
        "gene_b" : "snf2",
        "published_only" : False,
        "max_scl" : 9
    }

    res = requests.post('http://localhost:5000/gi_triplets', json=rp)
    if res.ok:
        d = res.json()
        print(d['rows'])
        #print(json.dumps(d['rows'], indent=4))
    else:
        print(res.reason)
def test_gi_pairs():

    rp = {
        "species_id" : 1,
        "threshold" : 0.9,
        "gene_a" : "snf1",
        "gene_b" : "",
        "published_only" : False,
        "page" : 0,
        "max_spl" : 9
    }

    res = requests.post('http://localhost:8090/gi_pairs', json=rp)
    if res.ok:
        d = res.json()
        print(d['rows'])
        #print(json.dumps(d['rows'], indent=4))
    else:
        print(res.reason)

def test_interpret():

    rp = {
        "gi_id" : 85
    }

    res = requests.post('http://localhost:5000/gi', json=rp)
    if res.ok:
        d = res.json()
        print(json.dumps(d, indent=4))
    else:
        print(res.reason)


def test_common_interactors():

    rp = {
        "gene_a" : "snf1",
        "gene_b" : "snf4",
        "gene_c" : "",
        "gene_d" : "",
        "threshold" : 0.9,
        "species_id" : 1,
        "published_only" : False,
        "max_spl" : 9
    }

    res = requests.post('http://localhost:5000/common_interactors', json=rp)
    if res.ok:
        d = res.json()
        print(json.dumps(d, indent=4))
        print(len(d))
    else:
        print(res.reason)
    
if __name__ == "__main__":
    main()
