import ast 
from pprint import pprint 
from pathlib import Path

def main():
    analyzer = Analyzer()
    for path in Path('.').rglob('*.py'):
        if path.parts[0] == 'venv':
            continue 
        if '.ipynb_checkpoints' in path.parts:
            continue
        print(path)

        with open(path, "r") as source:
            tree = ast.parse(source.read())

        analyzer.visit(tree)
    analyzer.report() 

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"import": set(), "from": set()}
        
    def visit_Import(self, node):
        for alias in node.names:
            self.stats["import"].add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.stats["from"].add(node.module)

        self.generic_visit(node)

    def report(self):
        pprint(self.stats['import'].union(self.stats['from']))

if __name__ == "__main__":
    main()
