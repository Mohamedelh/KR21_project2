from ast import Dict
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
        """ Simplifies a bayesian network for queries of the form Pr(Q|e)
        by pruning edges and nodes

        :param Q: a set of variables of which the probability needs to be computed
        :param e: a set of evidences that may affect the probability of Q
        """
        self._prune_edges(e)
        self._prune_nodes(Q, e)

    def _prune_edges(self, e: pd.Series) -> None:
        """ Prunes edges outgoing from nodes in e. It does so performing
        factor reduction on rows for a given node, that are not compatible with e

        :param e: a set of evidences that cause the pruning of edges
        """
        for variable in e.keys():
            # Update CPT of variable with reduced factor
            self.bn.update_cpt(variable, self.bn.get_compatible_instantiations_table(e, self.bn.get_cpt(variable)))

            descendants = self.bn.get_children(variable)
            for descendant in descendants:
                # Update CPT of descendant with reduced factor
                self.bn.update_cpt(descendant, self.bn.get_compatible_instantiations_table(e, self.bn.get_cpt(descendant)))

                # Remove edge from network
                self.bn.del_edge((variable, descendant))

    def _prune_nodes(self, Q: list[str], e: pd.Series) -> None:
        """ Prunes nodes by deleting leaf variables that are not in Q or e

        :param Q: a set of variables that cause the pruning of nodes
        :param e: a set of evidences, of which the variables cause the 
        pruning of nodes
        """
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
        """ Checks whether X and Y are D-separated by Z, by pruning the
        DAG and checking afterwards whether X and Y are connected or not

        :param X: a set of variables
        :param Y: a set of variables
        :param Z: a set of variables 
        :returns: True if X and Y are D-seperated, otherwise False
        """
        bn = deepcopy(self.bn)

        # Remove outging edges from Z
        for z in Z:
            descendants = self.bn.get_children(z)
            for descendant in descendants:
                bn.del_edge((z, descendant))

        for z in Z:
            print(self.bn.get_children(z))

        while True:
            leaf_variables = []
            for variable in self.bn.get_all_variables():
                if not self.bn.get_children(variable) and variable not in X + Y + Z:
                    leaf_variables.append(variable)
                    bn.del_var(variable)

            if not (leaf_variables):
                # Check if X and Y are connected through edges in a pruned network.
                # If not, then they are d-separated.
                return not all(any(nx.has_path(self.bn.structure, start, end) for end in Y) for start in X)      

    def independence(self, X: list[str], Y: list[str], Z: list[str]) -> bool:
        """ Checks whether X is independent of Y given Z, by checking
        whether X and Y are D-separated given Z

        :param X: a set of variables
        :param Y: a set of variables
        :param Z: a set of variables 
        :returns: True if X and Y are independent of each other, otherwise False
        """
        # Each d-separation implies an independence in a Bayesian network
        return self.d_separation(X, Y, Z) 

    def marginalization(self, X: str, cpt: pd.DataFrame) -> pd.DataFrame:
        """ Sums out variable X from a CPT

        :param X: a variable that is to be summed out
        :param cpt: the conditional probability table where X is to be 
        summed out
        :returns: a new CPT with X summed out
        """
        Y = self._get_variables_from_cpt(cpt)
        Y.remove(X)

        if not Y:
            # Empty set of variables, so only return p (Trival factor)
            new_cpt = pd.DataFrame()
            new_cpt['p'] = [sum(cpt['p'].tolist())]
            if 'Instantiations' in cpt:
                instantiations = {}
                for _, row in cpt.iterrows():
                    instantiations.update(row['Instantiations'])
                new_cpt['Instantiations'] = instantiations

            return new_cpt
            

        # Group by the remaining variables and sum
        return cpt.loc[:, ~cpt.columns.isin([X])].groupby(Y).sum().reset_index()

    def maxing_out(self, X: str, cpt: pd.DataFrame) -> pd.DataFrame:
        """ Maxes out variable X from a CPT

        :param X: a variable that is to be maxed out
        :param cpt: the conditional probability table where X is to be 
        maxed out
        :returns: a new CPT with X maxed out
        """
        # Exclude X and p from cpt
        Y = self._get_variables_from_cpt(cpt)        
        Y.remove(X)

        if not Y:
            # Empty set of variables, so only return p (Trival factor)
            new_cpt = pd.DataFrame()
            for _, row in cpt.iterrows():
                if 'p' not in new_cpt or row['p'] > new_cpt['p'].iloc[0]:
                    new_cpt['p'] = [row['p']]
                    if 'Instantiations' in cpt:
                        new_cpt['Instantiations'] = [{
                            X: row[X]
                        } | row['Instantiations']]
                    else:
                        new_cpt['Instantiations'] = [{
                            X: row[X]
                        }]
                
            return new_cpt

        # Group by the remaining variables and get max        
        # For each row in maxed result, check what instantiation of X led
        # to the maximized value and return it
        maxed_cpt = cpt.loc[cpt.groupby(Y)['p'].idxmax()]
        instantiations = []

        for _, row in maxed_cpt.iterrows():
            if 'Instantiations' in cpt:
                instantiations.append({
                    X: row[X]
                } | row['Instantiations'])
            else:
                instantiations.append({
                    X: row[X]
                })

        maxed_cpt['Instantiations'] = instantiations
 
        return maxed_cpt.drop([X], axis=1)
        
    def factor_multiplication(self, cpt_1: pd.DataFrame, cpt_2: pd.DataFrame) -> pd.DataFrame:
        """ Merges two CPT's by multiplying the rows with intersecting variables

        :param cpt_1: one of the CPT's that is to be merged
        :param cpt_2: one of the CPT's that is to be merged
        :returns: new CPT with both CPT's merged
        """
        # Get all variables from both
        Y = self._get_variables_from_cpt(cpt_1)
        Z = self._get_variables_from_cpt(cpt_2)

        if not Y and not Z:
            # Both are empty, meaning they are trivial factors
            new_cpt = pd.DataFrame()
            new_cpt['p'] = [cpt_1['p'].iloc[0] * cpt_2['p'].iloc[0]]

            if 'Instantiations' in cpt_1 or 'Instantiations' in cpt_2:
                new_instantiation = {}
                if 'Instantiations' in cpt_1:
                    new_instantiation = new_instantiation | cpt_1['Instantiations'].iloc[0]
                if 'Instantiations' in cpt_2:
                    new_instantiation = new_instantiation | cpt_2['Instantiations'].iloc[0]

            new_cpt['Instantiations'] = [new_instantiation]

            return new_cpt


        # Get intersected variables as they will decide what rows to multiply
        intersected = list(set(Y) & set(Z))
        if not intersected:
            raise ValueError("No intersected variable found")

        variables = list(dict.fromkeys(Y + Z))

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
        """ Returns an elimination order based on minimum amount of
        degrees (i.e. minimum amount of dependencies)

        :param X: a set of variables that is to be ordered
        :returns: an ordered list of variables
        """
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
        """ Returns an elimination order based on minimum amount of
        new interactions (i.e. new dependencies)

        :param X: a set of variables that is to be ordered
        :returns: an ordered list of variables
        """
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

    def variable_elimination(self, X: list[str], order: list[str], cpts: dict[str, pd.DataFrame] = None) -> dict[str, pd.DataFrame]:
        """ Eliminates a set of variables X from all or selected CPTs, using an 
        elimination order

        :param X: list of variables that are to be eliminated
        :param order: the elimination order
        :param cpts: the CPTs where the variables are to be eliminated
        :returns: all CPTs without variables X.
        """
        # Get all CPTs
        cpts = self.bn.get_all_cpts() if not cpts else cpts

        for variable in order:
            if variable in X:
                # Gather all CPTs that contain given variable
                cpts_to_merge = pd.DataFrame()
                label = variable
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
        
        return cpts

    def marginal_distribution(self, Q: list[str], e: pd.Series, order: list[str]) -> pd.DataFrame:
        """ Computes the distribution of Q given e 

        :param Q: a set of variables
        :param e: the evidence
        :param order: the elimination order
        :returns: the distribution of Q given e
        """
        # Get all cpts
        cpts = self.bn.get_all_cpts()

        # Gather all variables that need to be eliminated to compute the joint marginal
        variables_to_eliminate = []
        for variable in self.bn.get_all_variables():
            if variable not in Q:
                variables_to_eliminate.append(variable)

        # Reduce all factors with respect to e, if given
        if e.any():
            for variable in cpts.keys():
                cpts[variable] = self.bn.reduce_factor(e, cpts[variable])

        # Compute the distribution
        results = self.variable_elimination(variables_to_eliminate, order, cpts)
        distribution = pd.DataFrame()
        for key in list(results):
            if distribution.empty:
                distribution = cpts[key]
            else:
                distribution = self.factor_multiplication(distribution, cpts[key])

        if e.any():
            # Sum out Q to obtain Pr(e)
            pr_e = None
            for variable in Q:
                pr_e = self.marginalization(variable, pr_e) if pr_e else self.marginalization(variable, distribution) 

            pr_e = pr_e.iloc[0]['p']

            # Compute Pr(Q|e) through normalization
            distribution['p'] /= pr_e

            return distribution

        # No evidence, just return distribution
        return distribution

    def map(self, Q: list[str], e: pd.Series, order: list[str]) -> pd.DataFrame:
        """ Computes the most likely instantiations of Q given evidence e

        :param Q: a list of variables whose most likely instantiation needs to be computed
        :param e: the evidence
        :param order: the order used for elimination order
        :returns: the most likely instantiations in a CPT
        """
        # Get all cpts
        cpts = self.bn.get_all_cpts()

        # Gather all variables that need to be eliminated
        variables_to_eliminate = []
        for variable in self.bn.get_all_variables():
            if variable not in Q:
                variables_to_eliminate.append(variable)

        # If evidence, reduce factors with respect to e
        if e.any():
            # Reduce all factors with respect to e
            for variable in cpts.keys():
                cpts[variable] = self.bn.reduce_factor(e, cpts[variable])

        # Eliminate
        results = self.variable_elimination(variables_to_eliminate, order, cpts)
        
        # Max out Q according to order, to obtain most likely instances
        map = pd.DataFrame()
        max_out_var = True 

        for variable in order:
            if variable in Q:
                for key in list(results):
                    if variable in results[key]:
                        if map.empty:
                            map = results[key]
                        else:
                            try:
                                map = self.factor_multiplication(map, results[key])
                            except ValueError:
                                # Independent variables detected. Max out already
                                max_out_var = False
                                while bool(self._get_variables_from_cpt(map)):
                                    map = self.maxing_out(self._get_variables_from_cpt(map)[0], map)

                                while bool(self._get_variables_from_cpt(results[key])):
                                    results[key] = self.maxing_out(self._get_variables_from_cpt(results[key])[0], results[key])

                                map = self.factor_multiplication(map, results[key])

                        results.pop(key)
                if max_out_var:
                    map = self.maxing_out(variable, map)

        return map            

    def mpe(self, e: pd.Series, order: list[str]) -> pd.DataFrame:
        """ Computes the most likely instantiations of all
        variables except evidence.

        :param e: the evidence
        :param order: the order used for elimination order
        :returns: the most likely instantiations in a CPT
        """
        # Get all cpts and variables
        cpts = self.bn.get_all_cpts()
        variables = self.bn.get_all_variables()

        # Reduce all factors with respect to e
        for variable in variables:
            cpts[variable] = self.bn.reduce_factor(e, cpts[variable])

        # Max out according to order, to obtain most likely instances
        mpe = pd.DataFrame()
        max_out_var = True 

        for variable in order:
            for key in list(cpts):
                if variable in cpts[key]:
                    if mpe.empty:
                        mpe = cpts[key]
                    else:
                            try:
                                mpe = self.factor_multiplication(mpe, cpts[key])
                            except ValueError:
                                # Independent variables detected. Max out already
                                max_out_var = False
                                while bool(self._get_variables_from_cpt(mpe)):
                                    mpe = self.maxing_out(self._get_variables_from_cpt(mpe)[0], mpe)

                                while bool(self._get_variables_from_cpt(cpts[key])):
                                    cpts[key] = self.maxing_out(self._get_variables_from_cpt(cpts[key])[0], cpts[key])

                                mpe = self.factor_multiplication(mpe, cpts[key])

                    cpts.pop(key)

            if max_out_var:
                mpe = self.maxing_out(variable, mpe)

        return mpe   

    def _get_variables_from_cpt(self, cpt: pd.DataFrame) -> list[str]:
        return cpt.loc[:, ~cpt.columns.isin(['p', 'Instantiations'])].columns.tolist()

if __name__ == '__main__':
    bn_reasoner = BNReasoner('testing/test.BIFXML')
    merged_cpt = bn_reasoner.factor_multiplication(bn_reasoner.bn.get_cpt('A'), bn_reasoner.bn.get_cpt('B'))
    print(merged_cpt.loc[(merged_cpt['A'] == False) & (merged_cpt['B'] == False)])