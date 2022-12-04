import pandas as pd
from BNReasoner import BNReasoner

TEST_FILE = "testing/lecture_example.BIFXML"

def test_d_separation():
    pass

def test_marginalization():
    bn_reasoner = BNReasoner('testing/lecture_example.BIFXML')
    marginalized = bn_reasoner.marginalization('Rain?', bn_reasoner.bn.get_cpt('Slippery Road?'))

    assert marginalized.shape[1] == 2
    assert marginalized['Slippery Road?'].tolist() == [False, True] and marginalized['p'].tolist() == [1.3, 0.7]

# def test_prune_network():
#     bn_reasoner = BNReasoner('testing/lecture_example.BIFXML')
#     bn_reasoner.prune_network(
#         ['Wet Grass?'],
#         pd.Series({'Rain?': False, 'Winter?': True})
#     )

#     # Network check
#     assert bn_reasoner.bn.get_all_variables() == ['Winter?', 'Sprinkler?', 'Rain?', 'Wet Grass?']
#     assert bn_reasoner.bn.get_children('Winter?') == []
#     assert bn_reasoner.bn.get_children('Rain?') == []
#     assert bn_reasoner.bn.get_children('Sprinkler?') == ['Wet Grass?']
#     assert bn_reasoner.bn.get_children('Wet Grass?') == []

#     # CPT check
#     assert bn_reasoner.bn.get_cpt('Winter?')['p'].tolist() == [0.4, 0.6] and bn_reasoner.bn.get_cpt('Winter?')['Winter?'].tolist() == [False, True]
#     assert bn_reasoner.bn.get_cpt('Rain?')['p'].tolist() == [0, 0, 0.2, 0]
