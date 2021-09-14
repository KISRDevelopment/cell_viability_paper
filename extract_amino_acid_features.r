
#Load libraries

library("protr")
library("parallel")

# An mc-version of the sapply function.
# From: https://stackoverflow.com/questions/31050556/parallel-version-of-sapply
mcsapply <- function (X, FUN, ..., simplify = TRUE, USE.NAMES = TRUE) {
  FUN <- match.fun(FUN)
  answer <- parallel::mclapply(X = X, FUN = FUN, ...)
  if (USE.NAMES && is.character(X) && is.null(names(answer))) 
    names(answer) <- X
  if (!isFALSE(simplify) && length(answer)) 
    simplify2array(answer, higher = (simplify == "array"))
  else answer
}

# Read fasta file
genes = readFASTA("../data-sources/yeast/orf_trans_all_nostar.fasta")

# Eliminate seqs with stop codon in the middle
genes = genes[(sapply(genes, protcheck))]


#Filter out sequences < 50 amico acids
genes2 =c()
for (i in 1:length(genes)){
  if(nchar(genes[[i]])>=50)
    genes2=c(genes2,genes[i])
}
genes=genes2

# Extract features
print("extractAAC ..")
aac = t(mcsapply(genes, extractAAC, mc.cores = 32))
colnames(aac) <- paste("AAC", colnames(aac), sep = "_")

print("extractDC ..")
dc = t(mcsapply(genes, extractDC, mc.cores = 32))
colnames(dc) <- paste("DC", colnames(dc), sep = "_")

print("extractTC ..")
tc = t(mcsapply(genes, extractTC,  mc.cores = 32))
colnames(tc) <- paste("TC", colnames(tc), sep = "_")

print("extractCTriad ..")
ctriad = t(mcsapply(genes, extractCTriad, mc.cores = 32))
colnames(ctriad) <- paste("CTriad", colnames(ctriad), sep = "_")

print("extractCTDC ..")
ctdc = t(mcsapply(genes, extractCTDC, mc.cores = 32)) 
colnames(ctdc) <- paste("CTDC", colnames(ctdc), sep = "_")

print("extractCTDT ...")
ctdt = t(mcsapply(genes, extractCTDT, mc.cores = 32)) 
colnames(ctdt) <- paste("CTDT", colnames(ctdt), sep = "_")

print("extractCTDD ...")
ctdd = t(mcsapply(genes, extractCTDD, mc.cores = 32)) 
colnames(ctdd) <- paste("CTDD", colnames(ctdd), sep = "_")

print("extractMoreauBroto ...")
corr_mb = t(mcsapply(genes, extractMoreauBroto, mc.cores = 32)) 
colnames(corr_mb) <- paste("MB", colnames(corr_mb), sep = "_")

print("extractMoran ...")
corr_moran = t(mcsapply(genes, extractMoran, mc.cores = 32)) 
colnames(corr_moran) <- paste("MORAN", colnames(corr_moran), sep = "_")

print("extractGeary ...")
corr_geary = t(mcsapply(genes, extractGeary, mc.cores = 32))
colnames(corr_geary) <- paste("GEARY", colnames(corr_geary), sep = "_")

print("extractSOCN ...")
socn = t(mcsapply(genes, extractSOCN, mc.cores = 32))
colnames(socn) <- paste("SOCN", colnames(socn), sep = "_")

print("extractQSO ...")
qso = t(mcsapply(genes, extractQSO,nlag=30, mc.cores = 32))
colnames(qso) <- paste("QSO", colnames(qso), sep = "_")

print("extractPAAC ...")
paac = t(mcsapply(genes, extractPAAC,lambda=30, mc.cores = 32))
colnames(paac) <- paste("PAAC", colnames(paac), sep = "_")

print("extractAPAAC ...")
apaac = t(mcsapply(genes, extractAPAAC, mc.cores = 32,lambda=30))
colnames(apaac) <- paste("APAAC", colnames(apaac), sep = "_")

F = cbind(aac, dc, tc, ctriad, ctdc, ctdt, ctdd, corr_mb, corr_moran, corr_geary, socn, qso, paac, apaac)
save(F, file="../tmp/amino_acid_features.Rda")
