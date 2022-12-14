import pandas as pd
from BNReasoner import BNReasoner

TEST_FILE = 'testing/test.BIFXML'

def test_prune_bn():
    bn_reasoner = BNReasoner(TEST_FILE)
    Q = ['B']
    e = pd.Series({'A': True})

    bn_reasoner.prune_bn(Q, e)
    variables = bn_reasoner.bn.get_all_variables()
    table_A = bn_reasoner.bn.get_cpt('A')
    table_B = bn_reasoner.bn.get_cpt('B')

    assert sorted(variables) == sorted(['A', 'B'])
    assert round(table_A.loc[table_A['A'] == True]['p'].iloc[0], 1) == 0.6
    assert round(table_B.loc[table_B['B'] == False]['p'].iloc[0], 1) == 0.1 and round(table_B.loc[table_B['B'] == True]['p'].iloc[0], 1) == 0.9
    
def test_d_separation():
    bn_reasoner = BNReasoner(TEST_FILE)
    X = ['C']
    Y = ['A']
    Z = ['B']

    assert bn_reasoner.d_separation(X, Y, Z) == True

def test_independence():
    bn_reasoner = BNReasoner(TEST_FILE)
    X = ['C']
    Y = ['A']
    Z = ['B']

    assert bn_reasoner.independence(X, Y, Z) == True

def test_marginalization():
    bn_reasoner = BNReasoner(TEST_FILE)
    summed_out_A = bn_reasoner.marginalization('A', bn_reasoner.bn.get_cpt('B'))

    assert round(summed_out_A.loc[summed_out_A['B'] == True]['p'].iloc[0], 1) == 1.1 and round(summed_out_A.loc[summed_out_A['B'] == False]['p'].iloc[0], 1) == 0.9

def test_maxing_out():
    bn_reasoner = BNReasoner(TEST_FILE)
    maxed_out_A = bn_reasoner.maxing_out('A', bn_reasoner.bn.get_cpt('B'))

    assert round(maxed_out_A.loc[maxed_out_A['B'] == True]['p'].iloc[0], 1) == 0.9 and round(maxed_out_A.loc[maxed_out_A['B'] == False]['p'].iloc[0], 1) == 0.8
    assert maxed_out_A.loc[maxed_out_A['B'] == True]['Instantiations'].iloc[0]['A'] == True and maxed_out_A.loc[maxed_out_A['B'] == False]['Instantiations'].iloc[0]['A'] == False

def test_factor_multiplication():
    bn_reasoner = BNReasoner(TEST_FILE)
    merged_cpt = bn_reasoner.factor_multiplication(bn_reasoner.bn.get_cpt('A'), bn_reasoner.bn.get_cpt('B'))

    assert round(merged_cpt.loc[(merged_cpt['A'] == False) & (merged_cpt['B'] == False)]['p'].iloc[0], 2) == 0.32
    assert round(merged_cpt.loc[(merged_cpt['A'] == False) & (merged_cpt['B'] == True)]['p'].iloc[0], 2) == 0.08
    assert round(merged_cpt.loc[(merged_cpt['A'] == True) & (merged_cpt['B'] == False)]['p'].iloc[0], 2) == 0.06
    assert round(merged_cpt.loc[(merged_cpt['A'] == True) & (merged_cpt['B'] == True)]['p'].iloc[0], 2) == 0.54

def test_min_degree_ordering():
    bn_reasoner = BNReasoner(TEST_FILE)
    order = bn_reasoner.min_degree_ordering(['A', 'B', 'C'])

    assert order == ['A', 'B', 'C']

def test_min_fill_ordering():
    bn_reasoner = BNReasoner(TEST_FILE)
    order = bn_reasoner.min_fill_ordering(['A', 'B', 'C'])

    assert order == ['A', 'B', 'C']

def test_variable_elimination():
    bn_reasoner = BNReasoner(TEST_FILE)
    eliminated_cpt = bn_reasoner.variable_elimination(['A', 'C'], ['C', 'B', 'A'], {'B': bn_reasoner.bn.get_cpt('B'), 'C': bn_reasoner.bn.get_cpt('C')})
    eliminated_cpt = bn_reasoner.factor_multiplication(eliminated_cpt['C'], eliminated_cpt['AA'])

    assert round(eliminated_cpt.loc[(eliminated_cpt['B'] == True)]['p'].iloc[0], 1) == 1.1
    assert round(eliminated_cpt.loc[(eliminated_cpt['B'] == False)]['p'].iloc[0], 1) == 0.9

def test_marginal_distribution():
    bn_reasoner = BNReasoner(TEST_FILE)
    distribution = bn_reasoner.marginal_distribution(
        ['C'],
        pd.Series({
            'A': True
        }),
        ['C', 'B', 'A']
    )
    
    assert round(distribution.loc[(distribution['C'] == True)]['p'].iloc[0], 2) == 0.32
    assert round(distribution.loc[(distribution['C'] == False)]['p'].iloc[0], 2) == 0.68

def test_map():
    bn_reasoner = BNReasoner(TEST_FILE)
    map = bn_reasoner.map(
        ['A'],
        pd.Series({
            'C': True
        }),
        ['A', 'B', 'C']
    )

    assert map['Instantiations'].iloc[0]['A'] == True

def test_mpe():
    bn_reasoner = BNReasoner(TEST_FILE)
    mpe = bn_reasoner.mpe(
        pd.Series({
            'C': True
        }),
        ['A', 'B', 'C']
    )

    assert mpe['Instantiations'].iloc[0]['A'] == True and mpe['Instantiations'].iloc[0]['B'] == True
