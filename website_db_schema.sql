
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
    gene_name varchar not null,
    FOREIGN KEY (species_id) REFERENCES species(species_id),
    UNIQUE(gene_name)
);
CREATE INDEX gene_species ON genes(species_id);

DROP TABLE IF EXISTS genetic_interactions;
CREATE TABLE genetic_interactions (
    gi_id integer primary key autoincrement,
    species_id integer not null,
    gene_a_id integer not null,
    gene_b_id integer not null,
    observed boolean not null,
    observed_gi boolean not null,
    prob_gi float not null,
    FOREIGN KEY(gene_a_id) REFERENCES genes(gene_id),
    FOREIGN KEY(gene_b_id) REFERENCES genes(gene_id),
    FOREIGN KEY(species_id) REFERENCES species(species_id),
    UNIQUE(gene_a_id, gene_b_id)
);
CREATE INDEX gi_species ON genetic_interactions (species_id);
CREATE INDEX gi_first_gene ON genetic_interactions (gene_a_id);
CREATE INDEX gi_second_gene ON genetic_interactions (gene_b_id);
