import BNReasoner
import BayesNet
from BayesNet import BayesNet
from BNReasoner import BNReasoner
import time
import pandas as pd


# start = time.time()
# end = time.time()
# print(end - start)
bn = BayesNet()

files = ['testing/dog_problem.BIFXML', 'testing/lecture_example.BIFXML', 'testing/lecture_example2.BIFXML']

speeddict_mindegree = dict()
speeddict_minfill = dict()
speeddict_minfillmpedog =dict()
speeddict_minfillmpelec1 = dict()
speeddict_minfillmpelec2 = dict()
speeddict_minfillmapdog = dict()
speeddict_minfillmaplec1 = dict()
speeddict_minfillmaplec2 = dict()
speeddict_mindegreempedog = dict()
speeddict_mindegreempelec1 = dict()
speeddict_mindegreempelec2 = dict()
speeddict_mindegreemapdog =dict()
speeddict_mindegreemaplec1 =dict()
speeddict_mindegreemaplec2 =dict()
speeddict_unprunedlec1 = dict()
speeddict_prunedlec1 = dict()
speeddict_unprunedlec2 = dict()
speeddict_prunedlec2 =dict()
speeddict = [speeddict_mindegree, speeddict_minfill, speeddict_minfillmpedog, speeddict_minfillmpelec1, speeddict_minfillmpelec2, speeddict_minfillmapdog, speeddict_minfillmaplec1, speeddict_minfillmaplec2, speeddict_mindegreempedog, speeddict_mindegreempelec1, speeddict_mindegreempelec2, speeddict_mindegreemapdog, speeddict_mindegreemaplec1, speeddict_mindegreemaplec2, speeddict_unprunedlec1, speeddict_prunedlec1, speeddict_unprunedlec2, speeddict_prunedlec2] 
speeddictstr =  ['speeddict_mindegree', 'speeddict_minfill', 'speeddict_minfillmpedog', 'speeddict_minfillmpelec1', 'speeddict_minfillmpelec2', 'speeddict_minfillmapdog', 'speeddict_minfillmaplec1', 'speeddict_minfillmaplec2', 'speeddict_mindegreempedog', 'speeddict_mindegreempelec1', 'speeddict_mindegreempelec2', 'speeddict_mindegreemapdog', 'speeddict_mindegreemaplec1', 'speeddict_mindegreemaplec2', 'speeddict_unprunedlec1', 'speeddict_prunedlec1', 'speeddict_unprunedlec2', 'speeddict_prunedlec2'] 
for ele in files:
    net = BNReasoner(ele)
    variables = net.bn.get_all_variables()
    start = time.time()
    net.variable_elimination(variables, net.min_degree_ordering(variables))
    end = time.time()
    speed = end - start
    speeddict_mindegree[ele] = speed


for ele in files:
    net = BNReasoner(ele)
    variables = net.bn.get_all_variables()
    start = time.time()
    net.variable_elimination(variables, net.min_fill_ordering(variables))
    end = time.time()
    speed = end - start
    speeddict_minfill[ele] = speed


files = ['testing/dog_problem.BIFXML', 'testing/lecture_example.BIFXML', 'testing/lecture_example2.BIFXML']

# net = BNReasoner('testing/dog_problem.BIFXML')
# variables = net.bn.get_all_variables()
# start = time.time()
# net.variable_elimination(variables, net.min_fill_ordering(variables))
# net.mpe(pd.Series({'light-on': True, 'family-out': False}), net.min_fill_ordering(variables))
# end = time.time()
# speed = end - start
# speeddict_minfillmpedog['testing/dog_problem.BIFXML'] = speed

net = BNReasoner('testing/lecture_example.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.variable_elimination(variables, net.min_fill_ordering(variables))
net.mpe(pd.Series({'Rain?': True, 'Wet Grass?': True}), net.min_fill_ordering(variables))
end = time.time()
speed = end - start
speeddict_minfillmpelec1['testing/lecture_example.BIFXML'] = speed

net = BNReasoner('testing/lecture_example2.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.variable_elimination(variables, net.min_fill_ordering(variables))
net.mpe(pd.Series({'I': True, 'Y': True}), net.min_fill_ordering(variables))
end = time.time()
speed = end - start
speeddict_minfillmpelec2['testing/lecture_example2.BIFXML'] = speed



# net = BNReasoner('testing/dog_problem.BIFXML')
# variables = net.bn.get_all_variables()
# start = time.time()
# net.variable_elimination(variables, net.min_fill_ordering(variables))
# net.map(variables, pd.Series({'light-on': True, 'family-out': False}), net.min_fill_ordering(variables))
# end = time.time()
# speed = end - start
# speeddict_minfillmapdog['testing/dog_problem.BIFXML'] = speed

net = BNReasoner('testing/lecture_example.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.variable_elimination(variables, net.min_fill_ordering(variables))
net.map(variables, pd.Series({'Rain?': True, 'Wet Grass?': True}), net.min_fill_ordering(variables))
end = time.time()
speed = end - start
speeddict_minfillmaplec1['testing/lecture_example.BIFXML'] = speed

net = BNReasoner('testing/lecture_example2.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.variable_elimination(variables, net.min_fill_ordering(variables))
net.map(variables, pd.Series({'I': True, 'Y': True}), net.min_fill_ordering(variables))
end = time.time()
speed = end - start
speeddict_minfillmaplec2['testing/lecture_example2.BIFXML'] = speed


# net = BNReasoner('testing/dog_problem.BIFXML')
# variables = net.bn.get_all_variables()
# start = time.time()
# net.variable_elimination(variables, net.min_degree_ordering(variables))
# net.mpe(pd.Series({'light-on': True, 'family-out': False}), net.min_fill_ordering(variables))
# end = time.time()
# speed = end - start
# speeddict_mindegreempedog['testing/dog_problem.BIFXML'] = speed

net = BNReasoner('testing/lecture_example.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.variable_elimination(variables, net.min_degree_ordering(variables))
net.mpe(pd.Series({'Rain?': True, 'Wet Grass?': True}), net.min_fill_ordering(variables))
end = time.time()
speed = end - start
speeddict_mindegreempelec1['testing/lecture_example.BIFXML'] = speed

net = BNReasoner('testing/lecture_example2.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.variable_elimination(variables, net.min_degree_ordering(variables))
net.mpe(pd.Series({'I': True, 'Y': True}), net.min_fill_ordering(variables))
end = time.time()
speed = end - start
speeddict_mindegreempelec2['testing/lecture_example2.BIFXML'] = speed


# net = BNReasoner('testing/dog_problem.BIFXML')
# variables = net.bn.get_all_variables()
# start = time.time()
# net.variable_elimination(variables, net.min_degree_ordering(variables))
# net.map(variables, pd.Series({'bowel-problem': True, 'family-out': True}), net.min_fill_ordering(variables))
# end = time.time()
# speed = end - start
# speeddict_mindegreemapdog['testing/dog_problem.BIFXML'] = speed

net = BNReasoner('testing/lecture_example.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.variable_elimination(variables, net.min_degree_ordering(variables))
net.map(variables, pd.Series({'Rain?': True, 'Wet Grass?': True}), net.min_fill_ordering(variables))
end = time.time()
speed = end - start
speeddict_mindegreemaplec1['testing/lecture_example.BIFXML'] = speed

net = BNReasoner('testing/lecture_example2.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.variable_elimination(variables, net.min_degree_ordering(variables))
net.map(variables, pd.Series({'I': True, 'Y': True}), net.min_fill_ordering(variables))
end = time.time()
speed = end - start
speeddict_mindegreemaplec2['testing/lecture_example2.BIFXML'] = speed









net = BNReasoner('testing/lecture_example.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.prune_bn(variables, pd.Series({'Rain?': True, 'Wet Grass?': True}))
net.variable_elimination(variables, net.min_degree_ordering(variables))
end = time.time()
speed = end - start
speeddict_prunedlec1['testing/lecture_example.BIFXML'] = speed


net = BNReasoner('testing/lecture_example.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.variable_elimination(variables, net.min_degree_ordering(variables))
end = time.time()
speed = end - start
speeddict_unprunedlec1['testing/lecture_example.BIFXML'] = speed


net = BNReasoner('testing/lecture_example2.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.prune_bn(variables, pd.Series({'I': True, 'Y': True}))
net.variable_elimination(variables, net.min_degree_ordering(variables))
end = time.time()
speed = end - start
speeddict_prunedlec2['testing/lecture_example2.BIFXML'] = speed

net = BNReasoner('testing/lecture_example2.BIFXML')
variables = net.bn.get_all_variables()
start = time.time()
net.variable_elimination(variables, net.min_degree_ordering(variables))
end = time.time()
speed = end - start
speeddict_unprunedlec2['testing/lecture_example2.BIFXML'] = speed


for i,v in enumerate(speeddict):
    print(speeddictstr[i])
    print('==========')
    print(v)
    print('\n')
    