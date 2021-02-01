import sqlite3 
import json 
import numpy as np 
from contextlib import closing 

class DbLayer:

    def __init__(self, path, entries_per_page):

        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._entries_per_page = entries_per_page

    def close(self):
        self._conn.close()
    
    def get_gene(self, species_id, gene):
        if gene is None:
            return 
        
        gene = gene.lower().strip()
        if gene == '':
            return None 
        
        query = "SELECT * FROM genes WHERE species_id = :species_id AND (locus_tag = :gene OR common_name = :gene)"

        with closing(self._conn.cursor()) as c:
            c.execute(query, { "species_id" : species_id, "gene" : gene })
            row = c.fetchone()
            return dict(row) 

    def get_pairs(self, species_id, threshold, gene_a, gene_b, page, published_only=False):
        print("get_pairs(%d, %0.2f, %s, %s)" % (species_id, threshold, gene_a, gene_b))
        
        both_genes_clause = """
            AND ((g.gene_a_id = :gene_a_id AND g.gene_b_id = :gene_b_id) OR
                 (g.gene_a_id = :gene_b_id AND g.gene_b_id = :gene_a_id))
        """
        one_gene_clause = """
            AND ((g.gene_a_id = :gene_a_id OR g.gene_b_id = :gene_b_id) OR
                 (g.gene_a_id = :gene_b_id OR g.gene_b_id = :gene_a_id))
        """
        
        genes_clause = ""
        if gene_a and gene_b:
            genes_clause = both_genes_clause
        elif gene_a or gene_b:
            genes_clause = one_gene_clause
        else:
            return [], 0
        
        gene_a_row = self.get_gene(species_id, gene_a)
        gene_b_row = self.get_gene(species_id, gene_b)

        gene_a_id = -1 if not gene_a_row else gene_a_row['gene_id']
        gene_b_id = -1 if not gene_b_row else gene_b_row['gene_id']
        

        query = """
                SELECT  g.gi_id gi_id,
                            g.species_id species_id,
                            a.locus_tag gene_a_locus_tag, 
                            b.locus_tag gene_b_locus_tag,
                            a.common_name gene_a_common_name,
                            b.common_name gene_b_common_name, 
                            g.observed, 
                            g.observed_gi, 
                            g.prob_gi,
                            a.lid gene_a_lid,
                            b.lid gene_b_lid,
                            a.smf gene_a_smf,
                            b.smf gene_b_smf,
                            a.sgo_terms gene_a_sgo,
                            b.sgo_terms gene_b_sgo,
                            g.spl spl
                FROM    genetic_interactions g 
                JOIN    genes a on g.gene_a_id = a.gene_id 
                JOIN    genes b on g.gene_b_id = b.gene_id
                WHERE   (a.species_id = :species_id) AND
                        (g.prob_gi >= :threshold) AND
                        (NOT :published_only OR (g.observed = :published_only AND g.observed_gi = 1))
                        %s
                        ORDER BY g.prob_gi DESC
                LIMIT :entries_per_page OFFSET :offset
        """ % genes_clause
        
        count_query = """
                SELECT COUNT(g.gi_id) as n_rows
                FROM    genetic_interactions g 
                JOIN    genes a on g.gene_a_id = a.gene_id 
                JOIN    genes b on g.gene_b_id = b.gene_id
                WHERE   (a.species_id = :species_id) AND
                        (g.prob_gi >= :threshold) AND 
                        (NOT :published_only OR (g.observed = :published_only AND g.observed_gi = 1))
                        %s
        """ % genes_clause
        
        params = { 
            "species_id" : species_id, 
            "threshold" : threshold,
            "gene_a_id" : gene_a_id,
            "gene_b_id" : gene_b_id,
            "published_only" : published_only,
            "entries_per_page" : self._entries_per_page,
            "offset" : self._entries_per_page * page, 
        }
        with closing(self._conn.cursor()) as c:
            c.execute(query, params)
            rows = [dict(r) for r in c.fetchall()]

            c.execute(count_query, params)
            n_rows = c.fetchone()['n_rows']
            return rows, n_rows
    
    def get_gi(self, gi_id):
        query = """
                SELECT  g.gi_id gi_id,
                            g.species_id species_id,
                            a.locus_tag gene_a_locus_tag, 
                            b.locus_tag gene_b_locus_tag,
                            a.common_name gene_a_common_name,
                            b.common_name gene_b_common_name, 
                            g.observed, 
                            g.observed_gi, 
                            g.prob_gi,
                            a.lid gene_a_lid,
                            b.lid gene_b_lid,
                            a.smf gene_a_smf,
                            b.smf gene_b_smf,
                            a.sgo_terms gene_a_sgo,
                            b.sgo_terms gene_b_sgo,
                            g.spl spl
                FROM    genetic_interactions g 
                JOIN    genes a on g.gene_a_id = a.gene_id 
                JOIN    genes b on g.gene_b_id = b.gene_id
                WHERE   g.gi_id = :gi_id
        """

        pub_query = "SELECT p.identifier identifier FROM gi_pubs p WHERE p.gi_id = ?"

        with closing(self._conn.cursor()) as c:
            c.execute(query, { 
                "gi_id" : gi_id
            })
            row = c.fetchone()
            gi_row = dict(row) if row is not None else None
            if gi_row is not None:
                c.execute(pub_query, (gi_id,))
                rows = c.fetchall()
                gi_row['pubs'] = [dict(r) for r in rows]
            return gi_row
 
if __name__ == "__main__":

    layer = DbLayer('db.sqlite', 50)

    rows, _ = layer.get_pairs(3, 0.5, 'myc', '', 0)
    assert(len(rows) > 0)

    rows, _ = layer.get_pairs(3, 0.5, '', 'myc', 0)
    assert(len(rows) > 0)

    rows, _ = layer.get_pairs(3, 0.5, 'myc', 'a12m1', 0)
    assert(len(rows) == 1)
    
    rows, _ = layer.get_pairs(3, 0.5, 'a12m1', 'myc', 0)
    assert(len(rows) == 1)

    rows, _ = layer.get_pairs(3, 0.5, '', 'myc', 100)
    assert(len(rows) > 0)

    rows, _ = layer.get_pairs(3, 0.5, '', '', 100)
    assert(len(rows) == 0)

    row = layer.get_gi(3)
    assert(row is not None)

    row = layer.get_gi('adf')
    assert(row is None)

    rows, _ = layer.get_pairs(1, 0.5, '', 'snf1', 0)
    assert(len(rows) > 0)
    print(len(rows))

    rows, n_rows = layer.get_pairs(3, 0.5, 'myc', '', 0)
    assert(n_rows > 0)
    print(n_rows)

    rows, n_rows = layer.get_pairs(3, 0.5, 'myc', '', 0, True)
    assert(n_rows > 0)
    print(n_rows)
    print(rows)

    row = layer.get_gi(3)
    print(row)
