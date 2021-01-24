
DROP TABLE IF EXISTS species;
CREATE TABLE species (
    species_id integer primary key autoincrement,
    species_name varchar not null
);
INSERT INTO species(species_name) VALUES("S. cerevisiae");
INSERT INTO species(species_name) VALUES("S. pombe");
INSERT INTO species(species_name) VALUES("H. sapiens");
INSERT INTO species(species_name) VALUES("D. melanogaster");

DROP TABLE IF EXISTS genes;
CREATE TABLE genes (
    gene_id integer primary key autoincrement,
    species_id integer not null, 
    locus_tag varchar not null,
    common_name varchar not null,
    lid float not null,
    smf integer not null,
    sgo_terms blob not null,
    FOREIGN KEY (species_id) REFERENCES species(species_id),
    UNIQUE(locus_tag)
);
CREATE INDEX gene_species ON genes(species_id);
CREATE INDEX gene_locus_tag ON genes(locus_tag);
CREATE INDEX gene_common_name ON genes(common_name);

DROP TABLE IF EXISTS genetic_interactions;
CREATE TABLE genetic_interactions (
    gi_id integer primary key autoincrement,
    species_id integer not null,
    gene_a_id integer not null,
    gene_b_id integer not null,
    observed boolean not null,
    observed_gi boolean not null,
    prob_gi float not null,
    spl float not null,
    FOREIGN KEY(gene_a_id) REFERENCES genes(gene_id),
    FOREIGN KEY(gene_b_id) REFERENCES genes(gene_id),
    FOREIGN KEY(species_id) REFERENCES species(species_id),
    UNIQUE(gene_a_id, gene_b_id)
);
CREATE INDEX gi_species ON genetic_interactions (species_id);
CREATE INDEX gi_first_gene ON genetic_interactions (gene_a_id);
CREATE INDEX gi_second_gene ON genetic_interactions (gene_b_id);
