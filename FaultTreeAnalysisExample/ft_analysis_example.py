import pandas as pd
from FaultTreeAnalysis import FTAnalysis

#See results in ft_analysis_example_results.png

# Fault tree structure Dataframe (it have to be structured as showed in fault_tree_structure_db.png)
# dataframe structure is showed in fault_tree_structure.png; E is the stand-by system SB_1_1_E1_E2
ft_structure = pd.read_excel("ft_test_1.xlsx")

# Item parameters structure Dataframe (it have to be structured as showed in Items_param_df.png)
#Sb_1_1 mean stand-by system 1 functioning 1 in cold stand by
item_params = pd.read_excel(r"ft_item_param.xlsx")

time = 8000

ft_anal = FTAnalysis(time, ft_structure, item_params)


# Calculate gates and system reliability

gates_reliability, system_reliability = ft_anal.ft_rel()
print("\nSystem Reliability: ", system_reliability)

# Calculate gates and system unreliability

gates_unreliability, system_unreliability = ft_anal.ft_unrel()  
print("\nSystem Unreliability: ", system_unreliability) 

# Calculate gates and system availability 

gates_availability, system_availability = ft_anal.ft_availability()

print("\nSystem Availability: ", system_availability) 

# Calculate gates and system unavailability

gates_unavailability, system_unavailability = ft_anal.ft_unavailability()

print("\nSystem Unavailability: ", system_unavailability) 

# Calculate gates and system MCS with MOCUS 

minimum_cut_sets = ft_anal.cut_sets
print("\nMinumu Cut Sets: ", minimum_cut_sets)


# Fault Tree Importance Measures

# The algorithm calculate the following measures:

'''  
# 1 - The Marginal measure:
The Marginal measure, also called Birnbaum, measures the increase in the 
probability (P) of the top event (E) due to an event (A). It is reported as the
 difference in the probability of E given that A did occur (probability of 
event A is set to 1) and the probability of E given that A did not occur 
(probability of event A is set to 0).

Marginal Importance Measure = P(E|P(A)=1) – P(E|P(A)=0)

It allows you to see the increase in the probability of E given the occurrence 
of A.
One weakness of the Marginal importance measure is that it does not directly 
consider the probability of event A occurring, which means you can be led to 
assign high importance values to events that are very unlikely to occur and thus 
may be difficult to improve.

# 2 - Criticality measure:
The Criticality measure is a modification of the Marginal importance measure that
also takes into account the probability of event A. It takes the Marginal 
importance measure and multiplies it by the probability of A divided by the 
probability of E.

Criticality Importance Measure = Marginal Importance Measure * P(A) / P(E)

Because it also takes into account the end event occurrence, it is used to 
highlight events that lead to the top event occurring and are also more likely 
to occur and thus can reasonably be improved.

# 3 - Diagnostic measure:
The Diagnostic measure is the fraction of the top event (E) probability (P) 
that includes the event (A) occurring; or it is the probability that if the 
top gate occurred, the event occurred.

Diagnostic Importance Measure = P(A) * P(E|P(A)=1) / P(E)

# 4 -Risk Achievement Worth (RAW):
The Risk Achievement Worth (RAW), or Top Increase Sensitivity, measure is the 
increase in probability of top event E when event A is given to occur. 
It reports the ratio of the probability of E when event A is given to occur 
(probability of event A is set to 1) and the probability of E.

RAW Importance Measure = P(E|P(A)=1) / P(E)

Events with the largest RAW measure values have the largest impact on the 
probability of the top gate, P(E), therefore it shows where prevention areas 
should be focused to prevent top event failures.

# 4 -Risk Achievement Worth (RAW):
The Risk Reduction Worth (RRW), or Top Decrease Sensitivity, measure is the 
reduction in probability of top event E when event A is given to not occur. I
t reports the ratio of the probability of E and the probability of E when event 
A is given to not occur (probability of event A is set to 0).

RRW Importance Measure = P(E) / P(E|P(A)=0)

The RRW measure determines the maximum reduction in the top event probability 
if the event is improved.

'''

# Calculate Component Importance measures unavailability
# Return a pandas Dataframe

print("\nUnavailability Comp. Importance")
comp_importance_unavailability = ft_anal.comp_imp_unavailability()

# Calculate Component Importance measures reliability
# Return a pandas Dataframe

print("\nLUnreliability Comp. Importance")
comp_importance_reliability = ft_anal.comp_imp_unreliability()
