from copy import deepcopy
from typing import Union
import pandas as pd
import networkx as nx

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

    def prune_bn(self, Q: list[str], e: pd.Series) -> None:
        self._prune_edges(e)
        self._prune_nodes(Q, e)

    def _prune_edges(self, e: pd.Series) -> None:
        for variable in e.keys():
            descendants = self.bn.get_children(variable)
            for descendant in descendants:
                # Update CPT of descendant with reduced factor
                reduced_cpt = self.bn.reduce_factor(e, self.bn.get_cpt(descendant))
                self.bn.update_cpt(descendant, reduced_cpt)

                # Remove edge from network
                self.bn.del_edge((variable, descendant))

    def _prune_nodes(self, Q: list[str], e: pd.Series) -> None:
        while True:
            # Get all "leaf nodes" variables (thus do not have descendants)
            leaf_variables = []
            for variable in self.bn.get_all_variables():
                if not self.bn.get_children(variable) and variable not in Q and variable not in e.keys():
                    leaf_variables.append(variable)

            if not leaf_variables:
                return self.bn

            for variable in leaf_variables:
                self.bn.del_var(variable)   
    
    def d_separation(self, X: list[str], Y: list[str], Z: list[str]) -> bool:
        # Ensure network is pruned.
        evidence = {}

        for z in Z:
            evidence[z] = True    

        self.prune_bn(X + Y, pd.Series(evidence))

        # Check if X and Y are connected through edges in a pruned network.
        # If not, then they are d-separated.
        return not nx.has_path(self.bn.structure, X, Y)

    def independence(self, X: list[str], Y: list[str], Z: list[str]) -> bool:
        # Each d-separation implies an independence in a Bayesian network
        return self.d_separation(X, Y, Z) 

    def marginalization(self, X: str, cpt: pd.DataFrame) -> pd.DataFrame:
        Y = cpt.loc[:, ~cpt.columns.isin([X, 'p', 'Instantiations'])].columns.tolist()

        # Group by the remaining variables and sum
        return cpt.loc[:, ~cpt.columns.isin([X])].groupby(Y).sum().reset_index()

    def maxing_out(self, X: str, cpt: pd.DataFrame) -> pd.DataFrame:
        # Exclude X and p from cpt
        Y = cpt.loc[:, ~cpt.columns.isin([X, 'p', 'Instantiations'])].columns.tolist()

        # Group by the remaining variables and get max        
        # For each row in maxed result, check what instantiation of X led
        # to the maximized value and return it
        maxed_cpt = cpt.loc[:, ~cpt.columns.isin([X])].groupby(Y).max().reset_index()
        keys = list(maxed_cpt.columns.values)
        final_cpt = cpt[cpt.set_index(keys).index.isin(maxed_cpt.set_index(keys).index)].reset_index().drop(['index'], axis=1)
        instantiations = []
        for _, row in final_cpt.iterrows():
            if 'Instantiations' in final_cpt:
                instantiations.append({
                    X: row[X]
                } | row['Instantiations'])
            else:
                instantiations.append({
                    X: row[X]
                })

        if 'Instantiations' in final_cpt:
            final_cpt = final_cpt.drop(['Instantiations'], axis=1)

        final_cpt.insert(len(final_cpt.columns), 'Instantiations', instantiations)
 
        return final_cpt.drop([X], axis=1)
        
    def factor_multiplication(self, cpt_1: pd.DataFrame, cpt_2: pd.DataFrame) -> pd.DataFrame:
        # Get all variables from both
        Y = cpt_1.loc[:, ~cpt_1.columns.isin(['p', 'Instantiations'])].columns.tolist()
        Z = cpt_2.loc[:, ~cpt_2.columns.isin(['p', 'Instantiations'])].columns.tolist()

        # Get intersected variables as they will decide what rows to multiply
        variables = list(dict.fromkeys(Y + Z))
        intersected = list(set(Y) & set(Z))

        # Prepare data and create new CPT
        rows = {
            'p': []
        }
        if 'Instantiations' in cpt_1 or 'Instantiations' in cpt_2:
            rows['Instantiations'] = []

        for variable in variables:
            rows[variable] = []
        
        new_cpt = pd.DataFrame(columns=variables + ['p'])

        # Loop through one CPT, checking what exactly to multiply using the
        # intersected values
        for _, row in cpt_1.iterrows():
            for _, row_2 in cpt_2.iterrows():
                if all(row[variable] == row_2[variable] for variable in intersected):
                    rows['p'].append(row['p'] * row_2['p'])
                    if 'Instantiations' in cpt_1 or 'Instantiations' in cpt_2:
                        new_instantiation = {}
                        if 'Instantiations' in cpt_1:
                            new_instantiation = new_instantiation | row['Instantiations']
                        if 'Instantiations' in cpt_2:
                            new_instantiation = new_instantiation | row_2['Instantiations']
                        
                        rows['Instantiations'].append(new_instantiation)
                        

                    for variable in variables:
                        if variable in cpt_1:
                            rows[variable].append(row[variable])
                        else:
                            rows[variable].append(row_2[variable])

                    

        # Insert everything into new CPT and return
        for key in rows.keys():
            new_cpt[key] = rows[key]

        return new_cpt

    def min_degree_ordering(self) -> list[str]:
        bn = deepcopy(self.bn)
        ordering = []

        while bn.get_interaction_graph().nodes:
            nodes = bn.get_interaction_graph().nodes
            X = min(bn.get_interaction_graph().degree(nodes), key = lambda t: t[1])[0]            
            ordering.append(X)
            neighbors = [neighbor for neighbor in bn.get_interaction_graph().neighbors(X)]
            for neighbor in neighbors:
                for potential_neighbor in neighbors:
                    if neighbor != potential_neighbor and not bn.get_interaction_graph().has_edge(neighbor, potential_neighbor):
                        bn.add_edge((neighbor, potential_neighbor))

            bn.del_var(X)

        return ordering

    def min_fill_ordering(self) -> list[str]:
        bn = deepcopy(self.bn)
        ordering = []

        while bn.get_interaction_graph().nodes:
            nodes = bn.get_interaction_graph().nodes
            X = None
            X_n_new_edges = None
            X_edges_to_add = []

            for node in nodes:
                neighbors = [neighbor for neighbor in bn.get_interaction_graph().neighbors(node)]
                new_edges = 0 
                edges_to_add = []
                for neighbor in neighbors:
                    for potential_neighbor in neighbors:
                        if neighbor != potential_neighbor and not bn.get_interaction_graph().has_edge(neighbor, potential_neighbor) and (potential_neighbor, neighbor) not in edges_to_add:
                            new_edges += 1
                            edges_to_add.append((neighbor, potential_neighbor))

                if X is None or new_edges < X_n_new_edges:
                    X = node
                    X_n_new_edges = new_edges
                    X_edges_to_add = edges_to_add

            ordering.append(X)
            for edge in X_edges_to_add:
                bn.add_edge(edge)

            bn.del_var(X)

        return ordering

if __name__ == '__main__':
    bn_reasoner = BNReasoner('testing/lecture_example.BIFXML')
    print(bn_reasoner.min_fill_ordering())
    
