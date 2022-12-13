import pandas as pd
from BNReasoner import BNReasoner

def test_prune_bn():
    bn_reasoner = BNReasoner('testing/lecture_example.BIFXML')
    Q = ['Wet Grass?']
    e = pd.Series({'Winter?': True, 'Rain?': False})

    bn_reasoner.prune_bn(Q, e)

    assert sorted(bn_reasoner.bn.get_all_variables()) == sorted(['Winter?', 'Rain?', 'Sprinkler?', 'Wet Grass?'])
    assert sorted(bn_reasoner._get_variables_from_cpt(bn_reasoner.bn.get_cpt('Wet Grass?'))) == sorted(['Wet Grass?', 'Sprinkler?'])
    assert sorted(bn_reasoner._get_variables_from_cpt(bn_reasoner.bn.get_cpt('Sprinkler?'))) == sorted(['Sprinkler?'])
    assert sorted(bn_reasoner._get_variables_from_cpt(bn_reasoner.bn.get_cpt('Rain?'))) == sorted(['Rain?'])
    assert sorted(bn_reasoner._get_variables_from_cpt(bn_reasoner.bn.get_cpt('Winter?'))) == sorted(['Winter?'])
