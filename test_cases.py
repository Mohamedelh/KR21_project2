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
    assert table_A.loc[table_A['A'] == True]['p'].iloc[0] == 0.6
    assert table_B.loc[table_B['B'] == False]['p'].iloc[0] == 0.1 and table_B.loc[table_B['B'] == True]['p'].iloc[0] == 0.9
    
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
