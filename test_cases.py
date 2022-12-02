import pandas as pd
from BNReasoner import BNReasoner

def test_prune_network():
    bn_reasoner = BNReasoner('testing/lecture_example.BIFXML')
    bn_reasoner.prune_network(
        ['Wet Grass?'],
        pd.Series({'Rain?': False, 'Winter?': True})
    )

    # Network check
    assert bn_reasoner.bn.get_all_variables() == ['Winter?', 'Sprinkler?', 'Rain?', 'Wet Grass?']
    assert bn_reasoner.bn.get_children('Winter?') == []
    assert bn_reasoner.bn.get_children('Rain?') == []
    assert bn_reasoner.bn.get_children('Sprinkler?') == ['Wet Grass?']
    assert bn_reasoner.bn.get_children('Wet Grass?') == []

    # CPT check
    assert bn_reasoner.bn.get_cpt('Winter?')['p'].tolist() == [0.4, 0.6] and bn_reasoner.bn.get_cpt('Winter?')['Winter?'].tolist() == [False, True]
