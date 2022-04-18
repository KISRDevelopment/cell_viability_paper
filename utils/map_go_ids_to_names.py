import obonet
import json

def main():

    path = '../tools/go.obo'

    G = obonet.read_obo(path)

    id_to_name = {id_: data.get('name') for id_, data in G.nodes(data=True)}

    with open('../generated-data/go_ids_to_names.json', 'w') as f:
        json.dump(id_to_name, f, indent=4)

if __name__ == "__main__":
    main()
