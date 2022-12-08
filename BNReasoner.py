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
        bn = deepcopy(self.bn)

        while True:
            leaf_variables = []
            for variable in self.bn.get_all_variables():
                if not self.bn.get_children(variable) and variable not in X + Y + Z:
                    leaf_variables.append(variable)
                    bn.del_var(variable)

            edges = []
            for z in Z:
                descendants = self.bn.get_children(z)
                edges += descendants
                for descendant in descendants:
                    bn.del_edge((z, descendant))
            
            if not (leaf_variables or edges):
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

    def min_degree_ordering(self, X: list[str]) -> list[str]:
        interaction_graph = self.bn.get_interaction_graph()
        nodes = deepcopy(X)
        ordering = []

        while nodes:
            x = min(interaction_graph.degree(nodes), key = lambda t: t[1])[0]          
            ordering.append(x)
            neighbors = [neighbor for neighbor in interaction_graph.neighbors(x)]
            for neighbor in neighbors:
                for potential_neighbor in neighbors:
                    if neighbor != potential_neighbor and not interaction_graph.has_edge(neighbor, potential_neighbor):
                        interaction_graph.add_edge((neighbor, potential_neighbor))

            interaction_graph.remove_node(x)
            nodes.remove(x)

        return ordering

    def min_fill_ordering(self, X: list[str]) -> list[str]:
        interaction_graph = self.bn.get_interaction_graph()
        nodes = deepcopy(X)
        ordering = []

        while nodes:
            x = None
            x_n_new_edges = None
            x_edges_to_add = []

            for node in nodes:
                neighbors = [neighbor for neighbor in interaction_graph.neighbors(node)]
                new_edges = 0 
                edges_to_add = []
                for neighbor in neighbors:
                    for potential_neighbor in neighbors:
                        if neighbor != potential_neighbor and not interaction_graph.has_edge(neighbor, potential_neighbor) and (potential_neighbor, neighbor) not in edges_to_add:
                            new_edges += 1
                            edges_to_add.append((neighbor, potential_neighbor))

                if x is None or new_edges < x_n_new_edges:
                    x = node
                    x_n_new_edges = new_edges
                    x_edges_to_add = edges_to_add

            ordering.append(x)
            for edge in x_edges_to_add:
                interaction_graph.add_edge(edge)

            interaction_graph.remove_node(x)
            nodes.remove(x)

        return ordering

    def variable_elimination(self, X: list[str], heuristic: str = None) -> pd.DataFrame:
        # Get all CPTs
        cpts = self.bn.get_all_cpts()

        # Use order based on selected heuristic
        if heuristic == "min-degree":
            order = self.min_degree_ordering(X)
        elif heuristic == "min-fill":
            order = self.min_fill_ordering(X)
        else:
            order = X

        for variable in order:
            # Gather all CPTs that contain given variable
            cpts_to_merge = pd.DataFrame()
            label = ""
            for key in list(cpts):
                if variable in cpts[key]:
                    if cpts_to_merge.empty:
                        cpts_to_merge = cpts[key]
                    else:
                        cpts_to_merge = self.factor_multiplication(cpts_to_merge, cpts[key])

                    if variable != key:
                        label += key.replace(variable, "")
                    cpts.pop(key)
            
            # Sum out variable from merged cpt and add it to list of cpts
            if not cpts_to_merge.empty:
                cpts[label] = self.marginalization(variable, cpts_to_merge)

        final_cpt = pd.DataFrame()

        for key in list(cpts):
            if final_cpt.empty:
                final_cpt = cpts[key]
            else:
                final_cpt = self.factor_multiplication(final_cpt, cpts[key])

        return final_cpt

if __name__ == '__main__':
    bn_reasoner = BNReasoner('testing/lecture_example.BIFXML')
    cpts = bn_reasoner.bn.get_all_cpts()
    new_cpt = cpts['Winter?']

    for key in list(cpts):
        if key != 'Winter?':
            new_cpt = bn_reasoner.factor_multiplication(new_cpt, cpts[key])

    for i in ['Winter?', 'Rain?', 'Sprinkler?']:
        new_cpt = bn_reasoner.marginalization(i, new_cpt)

    print("EXPECTED")
    print(new_cpt)

    print("RECEIVED")
    print(bn_reasoner.variable_elimination(['Winter?', 'Rain?', 'Sprinkler?']))
    
