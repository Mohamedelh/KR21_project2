from typing import Union
import pandas as pd

from BayesNet import BayesNet



class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def prune_network(self, Q: list[str], e: pd.Series):
        self._prune_edges(e)
        self._prune_nodes(Q, e)

    def _prune_edges(self, e: pd.Series):
        for variable in e.keys():
            descendants = self.bn.get_children(variable)
            for descendant in descendants:
                # Update CPT of descendant with reduced factor
                reduced_cpt = self.bn.reduce_factor(e, self.bn.get_cpt(descendant))
                self.bn.update_cpt(descendant, reduced_cpt)

                # Remove edge from network
                self.bn.del_edge((variable, descendant))

    def _prune_nodes(self, Q: list[str], e: pd.Series):
        while True:
            # Get all "leaf nodes" variables (thus do not have descendants)
            leaf_variables = []
            for variable in self.bn.get_all_variables():
                if not self.bn.get_children(variable) and variable not in Q and variable not in e.keys():
                    print(variable)
                    leaf_variables.append(variable)

            if not leaf_variables:
                return

            for variable in leaf_variables:
                self.bn.del_var(variable)        
                