#
#   NameResolver processes the dataset of genome names 
#   located at: http://www.uniprot.org/docs/yeast
#   and allows the user to get locus tags, aliases, and both tag-alias
#
import os
import sys
import re
from collections import defaultdict

# regex to extract aliases and locus tags from each line in yeast.txt
pattern = re.compile(r'([^\s;]+)((?:; [^\s;]+)*)\s+([^\s]+)')

class NameResolver:

    def __init__(self, path='../data-sources/yeast/names.txt'):
        self._read_file(path)
        self._unidentified_names = set()
        
    def _read_file(self, path):
        
        self._alias_to_locus = {}
        self._locus_to_alias = defaultdict(list)
        
        with open(path, 'r') as f:
            
            for line in f:
                line = line.replace(' GAG ','').replace(' POL ','')
                
                match = pattern.match(line.lower())
                assert(match)
                
                aliases = [match.group(1)]
                tag = self.clean_up_name(match.group(3))
                
                # there is more than 1 alias, so extract them
                if match.group(2):
                    aliases = aliases + [e for e in match.group(2).split('; ') if e]
                
                # strict mode: only use primary alias
                for alias in aliases[:1]:
                    alias = self.clean_up_name(alias)

                    if alias in self._alias_to_locus:
                        print("Alias '%s' assigned to '%s' but new assignment is to '%s'" % (alias, self._alias_to_locus[alias], tag))
                    
                    self._alias_to_locus[alias] = tag
                    self._locus_to_alias[tag].append(alias)
        
    def get_unified_name(self, name):
        name = self.clean_up_name(name)
        self.unidentified = False

        if self.is_locus_name(name):
            return name + '  ' + self._locus_to_alias[name][0]
        elif self.is_common_name(name):
            locus_name = self._alias_to_locus[name]
            return locus_name + '  ' + self._locus_to_alias[locus_name][0]
        else:
            self._unidentified_names.add(name)
            self.unidentified = True
            return name + '  '
    
    def get_genes(self):
        return self.get_unified_names(self._locus_to_alias.keys())
        
    def clean_up_name(self, name):
        name = name.strip().lower()
        name = name.split('_damped')[0].strip()
        return name

    def get_unresolved_names(self):
        return self._unidentified_names
    
    def is_locus_name(self, name):
        return self.clean_up_name(name) in self._locus_to_alias
    
    def is_common_name(self, name):
        return self.clean_up_name(name) in self._alias_to_locus
    
    def get_locus_name(self, common_name):
        return self._alias_to_locus[common_name]
    
    def get_common_name(self, locus_name):
        return self._locus_to_alias[locus_name]
    
    def get_unified_names(self, names):
        return [self.get_unified_name(n) for n in names]
    
    def get_strain(self, locus_name):
        parts = locus_name.split('_')
        if len(parts) == 2:
            strain = re.sub(r'\d+', '', parts[1]).lower()
            return parts[0], strain
        return locus_name, ''
        
    def print_stats(self):

        num_locus_tags = len(self._locus_to_alias)
        num_aliases = len(self._alias_to_locus)

        alias_list_lens = [len(self._locus_to_alias[l]) for l in self._locus_to_alias]

        import numpy as np
        print("# locus tags: %d" % num_locus_tags)
        print("# aliases: %d" % num_aliases)
        print("# min alias: %d, max alias: %d, median: %0.2f" % (np.min(alias_list_lens),
            np.max(alias_list_lens), np.median(alias_list_lens)))

if __name__ == '__main__':
    
    resolver = NameResolver()
    
    resolver.print_stats()
