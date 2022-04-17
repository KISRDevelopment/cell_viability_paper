#
# creates the PPC network for all four organisms
#
import ppc_creation.ppc
import utils.map_entrez_fbgn
import os 

os.makedirs("../generated-data", exist_ok=True)

utils.map_entrez_fbgn.main()

ppc_creation.ppc.main("yeast", "../generated-data/ppc_yeast")
ppc_creation.ppc.main("pombe", "../generated-data/ppc_pombe")
ppc_creation.ppc.main("human", "../generated-data/ppc_human")
ppc_creation.ppc.main("dro", "../generated-data/ppc_dro")
