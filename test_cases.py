import pandas as pd
from BNReasoner import BNReasoner

def test_prune_bn():
    bn_reasoner = BNReasoner('testing/lecture_example.BIFXML')
    Q = ['Wet Grass?']
    e = pd.Series({'Winter?': True, 'Rain?': False})

    bn_reasoner.prune_bn(Q, e)

    assert sorted(bn_reasoner.bn.get_all_variables()) == sorted(['Winter?', 'Rain?', 'Sprinkler?', 'Wet Grass?'])