from BNReasoner import BNReasoner

if __name__ == '__main__':
    # Insert bayesian network file folder name here
    BAYESIAN_NETWORK_FOLDER = 'task_3.BIFXML'
    bn_reasoner = BNReasoner(BAYESIAN_NETWORK_FOLDER)

    # Here are the CPTs
    print("ALL CPTS")
    print(bn_reasoner.bn.get_all_cpts())

    # Define a Prior Marginal query here:
    print("PRIOR MARGINAL QUERY: ")
    Q = ['Variable1', 'Variable2']
    print(bn_reasoner.prior_marginal(Q, bn_reasoner.bn.get_all_variables()))


    # Define a posterior marginal query here:
    print("POSTERIOR MARGINAL QUERY: ")
    Q = ['AnotherVariableName']
    e = pd.Series({'VariableName': True})
    print(bn_reasoner.marginal_distribution(Q, e, bn_reasoner.bn.get_all_variables()))

    # Define a MAP query:
    print("MAP QUERY: ")
    Q = ['AnotherVariableName']
    e = pd.Series({'VariableName': True})
    print(bn_reasoner.map(Q, e, bn_reasoner.bn.get_all_variables()))

    # Define a MEP query:
    print("MEP QUERY: ")
    e = pd.Series({'VariableName': True})
    print(bn_reasoner.mpe(e, bn_reasoner.bn.get_all_variables()))
