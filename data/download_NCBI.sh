echo "Downloading data from NCBI"
mkdir -p raw
wget -O raw/NCBI.tsv \
https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Escherichia_coli_Shigella/PDG000000004.4059/Metadata/PDG000000004.4059.metadata.tsv