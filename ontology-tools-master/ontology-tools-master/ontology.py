import re
import itertools
import numpy as np
import networkx as nx
import networkx.algorithms.lowest_common_ancestors as nxlca
from lxml import etree


class Ontology(nx.DiGraph):
    def __init__(self):
        nx.DiGraph.__init__(self)

    def load(
        self,
        file,
        limit_prefixes=None,
        valid_relationships=["is_a", "part_of"],
    ):
        """Reads an .obo file into a networkx DiGraph object.

        Parameters
        ----------
        file : str
            Path to an .obo file containing the ontology.

        limit_prefixes : list(str)
            Specify allowed ontology term IDs, e.g. to load only CL and UBERON
            terms from the extended UBERON ontology, set
                `limit_prefixes = ["CL", "UBERON"]`
            If not set, will load all terms in the .obo file regardless of
            prefix.

            Default : `limit_prefixes = None`

        valid_relationships : list(str)
            Specify relationships that constitute an edge in the digraph.

            Default : `valid_relationships=["is_a", "part_of"]`
        """

        def allowed_ID(ID, limit_prefixes):
            if limit_prefixes is not None:
                return any(ID.startswith(prefix) for prefix in limit_prefixes)
            else:
                return True

        with open(file, "r") as f:
            text = f.read()

        terms = [
            term for term in text.split("\n\n") if term.startswith("[Term]")
        ]

        ID_pattern = re.compile(r"[A-Za-z]+:\d+")
        quote_pattern = re.compile(r'"(.*?)"')

        for term in terms:
            lines = term.split("\n")
            if lines[0] == "[Term]":
                lines = lines[1:]
            else:
                continue

            attributes = {}
            for line in lines:
                key, value = line.split(": ", maxsplit=1)

                if key == "is_a":
                    value = ID_pattern.match(value).group(0)
                elif key == "relationship":
                    key, value = value.split(" ! ")[0].split(" ")[:2]
                elif value == "true":
                    value = True
                elif value == "false":
                    value = False
                elif value.startswith('"'):
                    value = quote_pattern.match(value)[1]

                if key in attributes:
                    if not type(attributes[key]) == list:
                        attributes[key] = [attributes[key]]
                    attributes[key].append(value)
                else:
                    attributes[key] = value

            if attributes.get("is_obsolete"):
                continue  # skip obsolete terms

            if not allowed_ID(attributes["id"], limit_prefixes):
                continue  # skip if term does not have allowed prefix

            for key in attributes:
                if type(attributes[key]) == list:
                    # convert to hashable type
                    attributes[key] = tuple(attributes[key])

            self.add_node(attributes["id"], **attributes)

            for relationship in valid_relationships:
                if relationship in attributes:
                    if type(attributes[relationship]) == tuple:
                        for ID in attributes[relationship]:
                            if allowed_ID(ID, limit_prefixes):
                                self.add_edge(ID, attributes["id"])
                    else:
                        ID = attributes[relationship]
                        if allowed_ID(ID, limit_prefixes):
                            self.add_edge(ID, attributes["id"])

        if not nx.is_directed_acyclic_graph(self):
            raise nx.NetworkXException("graph is not acyclic")

    def ancestors(self, terms=None):
        """Returns the ancestors of given terms."""
        if type(terms) is str:
            return nx.ancestors(self, terms)
        else:
            return {term: nx.ancestors(self, term) for term in terms}

    def descendants(self, terms=None):
        """Returns the descendants of given terms."""
        if type(terms) is str:
            return nx.descendants(self, terms)
        else:
            return {term: nx.descendants(self, term) for term in terms}

    def similarity(self, term1, term2, measure="Resnik"):
        """Computes the semantic similarity between two terms."""
        if term1 == term2:
            lca = term1
        elif term1 in self.descendants(term2):
            lca = term2
        elif term2 in self.descendants(term1):
            lca = term1
        else:
            lca = nxlca.lowest_common_ancestor(self, term1, term2)

        if measure in ["Resnik", "Lin"]:
            if lca:
                c = (len(self.descendants(lca)) + 1) / len(self.nodes)
                similarity = -np.log(c)
                if measure == "Lin":
                    c1 = (len(self.descendants(term1)) + 1) / len(self.nodes)
                    c2 = (len(self.descendants(term2)) + 1) / len(self.nodes)
                    similarity = 2 * similarity / (-np.log(c1) - np.log(c2))
            else:
                similarity = 0
        else:
            raise (ValueError("measure={} is not supported".format(measure)))

        return similarity

    def similarities(self, terms=None, pairs=None, measure="Resnik"):
        """Computes semantic similarities between many terms or pairs of terms.
        More efficient than running `self.similarity` for each individual pair.
        """
        similarities = {}

        if terms is None:
            terms = self.nodes

        if pairs is None:
            pairs = itertools.combinations(terms, 2)

        descendants = self.descendants(self.nodes)
        pairs_lcas = nxlca.all_pairs_lowest_common_ancestor(self, pairs)

        if measure in ["Resnik", "Lin"]:
            for pair, lca in pairs_lcas:
                # check for networkx lca mistakes
                term1, term2 = pair
                if term1 == term2:
                    lca = term1
                elif term1 in descendants[term2]:
                    lca = term2
                elif term2 in descendants[term1]:
                    lca = term1

                if lca:
                    c = (len(descendants[lca]) + 1) / len(self.nodes)
                    similarities[pair] = -np.log(c)
                    if measure == "Lin":
                        c1 = (len(descendants[term1]) + 1) / len(self.nodes)
                        c2 = (len(descendants[term2]) + 1) / len(self.nodes)
                        similarities[pair] = (2 * similarities[pair]) / (
                            -np.log(c1) - np.log(c2)
                        )
                else:
                    similarities[pair] = 0
        else:
            raise (ValueError("measure={} is not supported".format(measure)))

        return similarities

    def write_edgelist(
        self, path, comments="#", delimiter="", data=True, encoding="utf-8"
    ):
        """Write graph as a list of edges.
        This is just a convenient shortcut to call nx.write_edgelist without
        having to import networkx."""
        nx.write_edgelist(
            self,
            path,
            comments=comments,
            delimiter=delimiter,
            data=data,
            encoding=encoding,
        )


class MeSH(Ontology):
    def __init__(self):
        Ontology.__init__(self)

    def load(self, file, limit_categories=None):
        """Reads a MeSH descriptor .xml file into a networkx DiGraph object.

        Parameters
        ----------
        file : str
            Path to an .xml file containing the MeSH structure.

        limit_categories : list(str)
            Specify allowed category IDs, e.g. to load only diseases and
            chemicals/drugs, set
                `limit_categories = ["C", "D"]`
            If not set, will load all terms in the MeSH hierarchy regardless
            of category. This typically results in multiple disconnected
            components. See https://meshb-prev.nlm.nih.gov/treeView for
            category IDs.

            Default : `limit_categories = None`
        """
        terms = etree.parse(file).getroot()
        ID_to_tree = {}
        tree_to_ID = {}

        # read basic node info and record structural information
        for term in terms:
            ID = term.findtext(".//DescriptorUI")
            name = term.find(".//DescriptorName").findtext(".//String")

            tree_numbers = term.find(".//TreeNumberList")
            if tree_numbers is not None:
                tree_numbers = [element.text for element in tree_numbers]

                # remove terms in categories that weren't requested
                if limit_categories:
                    tree_numbers = [
                        tree_number
                        for tree_number in tree_numbers
                        if tree_number.startswith(tuple(limit_categories))
                    ]

                # if the term has any tree structure information left,
                # add it to the graph
                if len(tree_numbers) > 0:
                    self.add_node(ID, name=name)
                    ID_to_tree[ID] = tree_numbers
                    for tree_number in ID_to_tree[ID]:
                        tree_to_ID[tree_number] = ID

        # parse hierarchical structure
        for ID, tree_numbers in ID_to_tree.items():
            for tree_number in tree_numbers:
                lineage = tree_number.split(".")

                # if the term has more than one section in its tree number,
                # it is a child term, so add parental relationships
                if len(lineage) > 1:
                    parent = ".".join(lineage[:-1])
                    self.add_edge(tree_to_ID[parent], ID)
