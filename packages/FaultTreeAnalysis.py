import numpy as np
import scipy.stats as st
import cutsets
import copy
import pandas as pd


# Stand-by 2 working 1 Stand-by availability/unavaiability

def stand_by_2_1_av(t_end, ua, ub, uc, la, lb, lc): 

    Dt = 0.001
    P1_start = 1
    P2_start = 0
    P3_start = 0
    P4_start = 0
    P5_start = 0
    P6_start = 0
    P7_start = 0
    P8_start = 0
    P9_start = 0
    P10_start = 0
    
    t_start = 0
 
    
    n_steps = int(round(t_end-t_start)/Dt)
    
    P1_arr = np.zeros(n_steps+1)
    P2_arr = np.zeros(n_steps+1)
    P3_arr = np.zeros(n_steps+1)
    P4_arr = np.zeros(n_steps+1)
    P5_arr = np.zeros(n_steps+1)
    P6_arr = np.zeros(n_steps+1)
    P7_arr = np.zeros(n_steps+1)
    P8_arr = np.zeros(n_steps+1)
    P9_arr = np.zeros(n_steps+1)
    P10_arr = np.zeros(n_steps+1)
    
    t_array = np.zeros(n_steps+1)
    
    P1_arr[0] = P1_start
    P2_arr[0] = P2_start
    P3_arr[0] = P3_start
    P4_arr[0] = P4_start
    P5_arr[0] = P5_start
    P6_arr[0] = P6_start
    P7_arr[0] = P7_start
    P8_arr[0] = P8_start
    P9_arr[0] = P9_start
    P10_arr[0] = P10_start
    
    t_array[0] = t_start
    
    
    
    for i in range(1, n_steps+1):
        
        P1 = P1_arr[i-1]
        P2 = P2_arr[i-1]
        P3 = P3_arr[i-1]
        P4 = P4_arr[i-1]
        P5 = P5_arr[i-1]
        P6 = P6_arr[i-1]
        P7 = P7_arr[i-1]
        P8 = P8_arr[i-1]
        P9 = P9_arr[i-1]
        P10 = P10_arr[i-1]
        
        t = t_array[i-1]
        
        dP1dt = uc*P6 - la*P1
        dP2dt = ua*P4 - lb*P2
        dP3dt = ub*P5 - lc*P3
        dP4dt = la*P1 + ub*P7 - (ua + lb)*P4
        dP5dt = lb*P2 + uc*P8 - (ub + lc)*P5
        dP6dt = lc*P3 + ua*P9 - (uc + la)*P6
        dP7dt = lb*P4 + uc*P10 - (ub + lc)*P7
        dP8dt = lb*P5 + ua*P10 - (uc + la)*P8
        dP9dt = lb*P6 + ub*P10 - (ua + lb)*P9
        dP10dt = lc*P7 + la*P8 + lb*P9 - (ua + ub + uc)*P10
        
        P1_arr[i] = P1 + Dt*dP1dt
        P2_arr[i] = P2 + Dt*dP2dt
        P3_arr[i] = P3 + Dt*dP3dt
        P4_arr[i] = P4 + Dt*dP4dt
        P5_arr[i] = P5 + Dt*dP5dt
        P6_arr[i] = P6 + Dt*dP6dt
        P7_arr[i] = P7 + Dt*dP7dt
        P8_arr[i] = P8 + Dt*dP8dt
        P9_arr[i] = P9 + Dt*dP9dt
        P10_arr[i] = P10 + Dt*dP10dt
        
        t_array[i] = t + Dt
    
    availability = P1+P2+P3+P4+P5+P6
    unavailability = P7+P8+P9+P10
    
    return availability, unavailability

# Stand-by 2 working 1 Stand-by reliability/unreliability


def stand_by_2_1_rel(t_end, la, lb, lc): 

    Dt = 0.001
    P1_start = 1
    P2_start = 0
    P3_start = 0
    P4_start = 0
    P5_start = 0
    P6_start = 0
    P7_start = 0

    
    t_start = 0
 
    
    n_steps = int(round(t_end-t_start)/Dt)
    
    P1_arr = np.zeros(n_steps+1)
    P2_arr = np.zeros(n_steps+1)
    P3_arr = np.zeros(n_steps+1)
    P4_arr = np.zeros(n_steps+1)
    P5_arr = np.zeros(n_steps+1)
    P6_arr = np.zeros(n_steps+1)
    P7_arr = np.zeros(n_steps+1)

    
    t_array = np.zeros(n_steps+1)
    
    P1_arr[0] = P1_start
    P2_arr[0] = P2_start
    P3_arr[0] = P3_start
    P4_arr[0] = P4_start
    P5_arr[0] = P5_start
    P6_arr[0] = P6_start
    P7_arr[0] = P7_start

    
    t_array[0] = t_start
    
    
    
    for i in range(1, n_steps+1):
        
        P1 = P1_arr[i-1]
        P2 = P2_arr[i-1]
        P3 = P3_arr[i-1]
        P4 = P4_arr[i-1]
        P5 = P5_arr[i-1]
        P6 = P6_arr[i-1]
        P7 = P7_arr[i-1]

        
        t = t_array[i-1]
        
        dP1dt = - (la + lb)*P1
        dP2dt = la*P1 - (lb + lc)*P2
        dP3dt = lb*P1 - (la + lc)*P3
        dP4dt = lb*P2 - lc*P4
        dP5dt = lc*P2 + la*P3 - lb*P5
        dP6dt = lc*P3 - la*P6
        dP7dt = la*P4 + lb*P5 + lc*P6

        
        P1_arr[i] = P1 + Dt*dP1dt
        P2_arr[i] = P2 + Dt*dP2dt
        P3_arr[i] = P3 + Dt*dP3dt
        P4_arr[i] = P4 + Dt*dP4dt
        P5_arr[i] = P5 + Dt*dP5dt
        P6_arr[i] = P6 + Dt*dP6dt
        P7_arr[i] = P7 + Dt*dP7dt

        
        t_array[i] = t + Dt
    
    reliability = P1+P2+P3+P4+P5+P6
    unreliability = P7
    
    return reliability, unreliability

# Stand-by 1 working 1 Stand-by availability/unavaiability

def sb_1_1_av_unav(ua, ub, la, lb, t_end):

    Dt = 0.001
    P1_start = 1
    P2_start = 0
    P3_start = 0
    P4_start = 0
    P5_start = 0
    
    t_start = 0
    
    n_steps = int(round(t_end-t_start)/Dt)
    
    P1_arr = np.zeros(n_steps+1)
    P2_arr = np.zeros(n_steps+1)
    P3_arr = np.zeros(n_steps+1)
    P4_arr = np.zeros(n_steps+1)
    P5_arr = np.zeros(n_steps+1)
    
    t_array = np.zeros(n_steps+1)
    
    P1_arr[0] = P1_start
    P2_arr[0] = P2_start
    P3_arr[0] = P3_start
    P4_arr[0] = P4_start
    P5_arr[0] = P5_start
    
    
    t_array[0] = t_start
     
    for i in range(1, n_steps+1):
    
        P1 = P1_arr[i-1]
        P2 = P2_arr[i-1]
        P3 = P3_arr[i-1]
        P4 = P4_arr[i-1]
        P5 = P5_arr[i-1]
        
        
        t = t_array[i-1]
        
        dP1dt = ub*P3 - la*P1
        dP2dt = ua*P4 - lb*P2
        dP3dt = lb*P2 + ua*P5 - (ub + la)*P3
        dP4dt = la*P1 + ub*P5 - (ua + lb)*P4
        dP5dt = la*P3 + lb*P4 - (ub + ub)*P5
        
        
        P1_arr[i] = P1 + Dt*dP1dt
        P2_arr[i] = P2 + Dt*dP2dt
        P3_arr[i] = P3 + Dt*dP3dt
        P4_arr[i] = P4 + Dt*dP4dt
        P5_arr[i] = P5 + Dt*dP5dt
        
        
        t_array[i] = t + Dt
        
    availability = P1+P2+P3+P4
    unavailability = P5
    
    return availability, unavailability

def sb_1_1_rel_un(la, lb, t_end):

    Dt = 0.001
    P1_start = 1
    P2_start = 0
    P3_start = 0

    
    t_start = 0
    
    n_steps = int(round(t_end-t_start)/Dt)
    
    P1_arr = np.zeros(n_steps+1)
    P2_arr = np.zeros(n_steps+1)
    P3_arr = np.zeros(n_steps+1)

    
    t_array = np.zeros(n_steps+1)
    
    P1_arr[0] = P1_start
    P2_arr[0] = P2_start
    P3_arr[0] = P3_start

    
    
    t_array[0] = t_start
     
    for i in range(1, n_steps+1):
    
        P1 = P1_arr[i-1]
        P2 = P2_arr[i-1]
        P3 = P3_arr[i-1]

        
        t = t_array[i-1]
        
        dP1dt = - la*P1
        dP2dt = la*P1 - lb*P2
        dP3dt = lb*P2

        
        
        P1_arr[i] = P1 + Dt*dP1dt
        P2_arr[i] = P2 + Dt*dP2dt
        P3_arr[i] = P3 + Dt*dP3dt

        
        
        t_array[i] = t + Dt
        
    availability = P1+P2
    unavailability = P3
    
    return availability, unavailability

# Single item availability

def availability(time_array, fail_rate, rep_rate):
    
    return (rep_rate/(fail_rate+rep_rate) + (fail_rate/(fail_rate+rep_rate))*(np.exp(-(fail_rate+rep_rate)*time_array)))
    
# Single item unavailability

def unavailability(time_array, fail_rate, rep_rate):
    
    return (fail_rate/(fail_rate+rep_rate))*(1-np.exp(-(fail_rate+rep_rate)*time_array))

# Items reliability calculation

def items_rel(time_array, item_params):  

    item = []
    qi = []

    for i in range(len(item_params)):
        
        distribution = item_params.loc[item_params.index==i, "distribution"][i]
        
        
        if distribution == "Weibull":         
            
            shape_param = item_params.loc[item_params.index==i, "shape_parameter"][i]
            scale_param = item_params.loc[item_params.index==i, "scale_parameter"][i]
            
            rel_f = 1-st.weibull_min.cdf(time_array, 
                                         shape_param, 
                                         scale = scale_param)
            
            item.append(item_params.loc[item_params.index==i, "Item"][i])
            qi.append(rel_f)
            
        elif distribution == "Normal":
            
            shape_param = item_params.loc[item_params.index==i, "shape_parameter"][i]
            scale_param = item_params.loc[item_params.index==i, "scale_parameter"][i]
            
            rel_f = 1-st.norm.cdf(time_array, 
                                  shape_param, 
                                  scale_param)
            
            item.append(item_params.loc[item_params.index==i, "Item"][i])
            qi.append(rel_f)        
            
        
        elif distribution == "Exponential":    

            scale_param = item_params.loc[item_params.index==i, "scale_parameter"][i]        
            
            rel_f = np.exp(-scale_param*time_array)
            
            item.append(item_params.loc[item_params.index==i, "Item"][i])
            qi.append(rel_f)
        
        elif distribution == "Lognormal":
            
            shape_param = item_params.loc[item_params.index==i, "shape_parameter"][i]
            scale_param = item_params.loc[item_params.index==i, "scale_parameter"][i]
        
            rel_f = 1-st.lognorm.cdf(time_array, 
                                     scale = scale_param)
            
            item.append(item_params.loc[item_params.index==i, "Item"][i])
            qi.append(rel_f)
            
        elif distribution == "Sb_1_1":
            
            shape_param = eval(item_params.loc[item_params.index==i, "failure_rate"][i])
            
            fail_rate_1 = shape_param[0]
            fail_rate_2 = shape_param[1]
            
            rel, unrel = sb_1_1_rel_un(fail_rate_1, fail_rate_2, time_array)
            
            Item_1 = eval(item_params.loc[item_params.index==i, "Item"][i])[0]
            Item_2 = eval(item_params.loc[item_params.index==i, "Item"][i])[1]
            
            item.append("SB_1_1_"+Item_1+"_"+Item_2)
            qi.append(rel)
            
    return dict(zip(item, qi))

def items_unrel(time_array, item_params): 
    
    item = []
    qi = []

    for i in range(len(item_params)):
        
        distribution = item_params.loc[item_params.index==i, "distribution"][i]
        
        
        if distribution == "Weibull":         
            
            shape_param = item_params.loc[item_params.index==i, "shape_parameter"][i]
            scale_param = item_params.loc[item_params.index==i, "scale_parameter"][i]
            
            rel_f = st.weibull_min.cdf(time_array, 
                                         shape_param, 
                                         scale = scale_param)
            
            item.append(item_params.loc[item_params.index==i, "Item"][i])
            qi.append(rel_f)
            
        elif distribution == "Normal":
            
            shape_param = item_params.loc[item_params.index==i, "shape_parameter"][i]
            scale_param = item_params.loc[item_params.index==i, "scale_parameter"][i]
            
            rel_f = st.norm.cdf(time_array, 
                                  shape_param, 
                                  scale_param)
            
            item.append(item_params.loc[item_params.index==i, "Item"][i])
            qi.append(rel_f)        
            
        
        elif distribution == "Exponential":    

            scale_param = item_params.loc[item_params.index==i, "scale_parameter"][i]        
            
            rel_f = 1-np.exp(-scale_param*time_array)
            
            item.append(item_params.loc[item_params.index==i, "Item"][i])
            qi.append(rel_f)
        
        elif distribution == "Lognormal":
            
            shape_param = item_params.loc[item_params.index==i, "shape_parameter"][i]
            scale_param = item_params.loc[item_params.index==i, "scale_parameter"][i]
        
            rel_f = st.lognorm.cdf(time_array, 
                                     scale = scale_param)
            
            item.append(item_params.loc[item_params.index==i, "Item"][i])
            qi.append(rel_f)
            
        elif distribution == "Sb_1_1":
            
            shape_param = eval(item_params.loc[item_params.index==i, "failure_rate"][i])
            
            fail_rate_1 = shape_param[0]
            fail_rate_2 = shape_param[1]
            
            rel, unrel = sb_1_1_rel_un(fail_rate_1, fail_rate_2, time_array)
            
            Item_1 = eval(item_params.loc[item_params.index==i, "Item"][i])[0]
            Item_2 = eval(item_params.loc[item_params.index==i, "Item"][i])[1]
            
            item.append("SB_1_1_"+Item_1+"_"+Item_2)
            qi.append(unrel)
            
    return dict(zip(item, qi))

    
def items_av(time_array, item_params):

    item = []
    qi = []

    for i in range(len(item_params)):
        
        distribution = item_params.loc[item_params.index==i, "distribution"][i]
        
            
        if distribution == "Sb_1_1":
            
            failure_par = eval(item_params.loc[item_params.index==i, "failure_rate"][i])
            repair_par = eval(item_params.loc[item_params.index==i, "repair_rate"][i])
            
            fail_rate_1 = failure_par[0]
            fail_rate_2 = failure_par[1]
            
            rep_rate_1 = repair_par[0]
            rep_rate_2 = repair_par[1]
            
            av, un = sb_1_1_av_unav(rep_rate_1, rep_rate_2, fail_rate_1, fail_rate_2, time_array)
            
            Item_1 = eval(item_params.loc[item_params.index==i, "Item"][i])[0]
            Item_2 = eval(item_params.loc[item_params.index==i, "Item"][i])[1]
            
            item.append("SB_1_1_"+Item_1+"_"+Item_2)
            qi.append(av)
            
        else:
            
            fail_rate = item_params.loc[item_params.index==i, "failure_rate"][i]
            rep_rate = item_params.loc[item_params.index==i, "repair_rate"][i]
            
            av = availability(time_array, fail_rate, rep_rate)
            
            item.append(item_params.loc[item_params.index==i, "Item"][i])
            qi.append(av)                
            
    return dict(zip(item, qi))


def items_unav(time_array, item_params):   

    item = []
    qi = []
    
    for i in range(len(item_params)):
        
        distribution = item_params.loc[item_params.index==i, "distribution"][i]
        
            
        if distribution == "Sb_1_1":
            
            failure_par = eval(item_params.loc[item_params.index==i, "failure_rate"][i])
            repair_par = eval(item_params.loc[item_params.index==i, "repair_rate"][i])
            
            fail_rate_1 = failure_par[0]
            fail_rate_2 = failure_par[1]
            
            rep_rate_1 = repair_par[0]
            rep_rate_2 = repair_par[1]
            
            av, un = sb_1_1_av_unav(rep_rate_1, rep_rate_2, fail_rate_1, fail_rate_2, time_array)
            
            Item_1 = eval(item_params.loc[item_params.index==i, "Item"][i])[0]
            Item_2 = eval(item_params.loc[item_params.index==i, "Item"][i])[1]
            
            item.append("SB_1_1_"+Item_1+"_"+Item_2)
            qi.append(un)
            
        else:
            
            fail_rate = item_params.loc[item_params.index==i, "failure_rate"][i]
            rep_rate = item_params.loc[item_params.index==i, "repair_rate"][i]
            
            unav = unavailability(time_array, fail_rate, rep_rate)
            
            item.append(item_params.loc[item_params.index==i, "Item"][i])
            qi.append(unav)                
            
    return dict(zip(item, qi))




class FaultTreeAnalysis:
    
    def __init__(self, time_array, ft_structure, item_params):        
      
        self.time_array = time_array
        self.ft_structure = ft_structure
        self.item_params = item_params
        
        ft = []

        for lev in list(ft_structure.columns):
            
            level = ft_structure[lev].dropna().apply(eval)
            
            for event in level:
                
                ft.append(tuple(event))
                
        self.cut_sets = cutsets.mocus(ft)
        
    def ft_rel(self):   
        
        
        Items_rel =  items_rel(self.time_array, self.item_params)
        
        for lev in list(self.ft_structure.columns)[::-1]:
            
            gates = self.ft_structure[lev].dropna().apply(eval)
            
            for gate in gates:
                
                
                if gate[1] == 'And':
                    
                    q = 1
                    
                    for event in gate[2]:
                        
                        q *= (1-Items_rel[event])
                        
                    q = 1-q            
        
                    
                elif gate[1] == 'Or':
            
                    q = 1
                    
                    for event in gate[2]:
                        
                        q *= Items_rel[event]             
        
                
                Items_rel[gate[0]] = q
                
        return Items_rel, Items_rel["TOP"]
    
    
   
    def ft_unrel(self):   
        

        Items_rel = items_unrel(self.time_array, self.item_params)
        
        for lev in list(self.ft_structure.columns)[::-1]:
            
            gates = self.ft_structure[lev].dropna().apply(eval)
            
            for gate in gates:
                
                
                if gate[1] == 'And':
     
                    q = 1
                    
                    for event in gate[2]:
                        
                        q *= Items_rel[event]       
                    
                
                elif gate[1] == 'Or':

                    q = 1
                    
                    for event in gate[2]:
                        
                        q *= (1-Items_rel[event])
                        
                    q = 1-q                 
                  
                Items_rel[gate[0]] = q
                
                       
        return Items_rel, Items_rel["TOP"]
    
    
    def ft_availability(self):   
        
        Items_rel = items_av(self.time_array, self.item_params)
        
        for lev in list(self.ft_structure.columns)[::-1]:
            
            gates = self.ft_structure[lev].dropna().apply(eval)
            
            for gate in gates:
                
                
                if gate[1] == 'And':
                    
                    q = 1
                    
                    for event in gate[2]:
                        
                        q *= (1-Items_rel[event])
                        
                    q = 1-q            
        
                    
                elif gate[1] == 'Or':
            
                    q = 1
                    
                    for event in gate[2]:
                        
                        q *= Items_rel[event]             
        
                
                Items_rel[gate[0]] = q
                
        return Items_rel, Items_rel["TOP"]
    
    
    def ft_unavailability(self):  
        
        Items_rel = items_unav(self.time_array, self.item_params)       
        
        
        for lev in list(self.ft_structure.columns)[::-1]:
            
            gates = self.ft_structure[lev].dropna().apply(eval)
            
            for gate in gates:
                
                
                if gate[1] == 'And':
     
                    q = 1
                    
                    for event in gate[2]:
                        
                        q *= Items_rel[event]       
                    
                
                elif gate[1] == 'Or':

                    q = 1
                    
                    for event in gate[2]:
                        
                        q *= (1-Items_rel[event])
                        
                    q = 1-q                 
                  
                Items_rel[gate[0]] = q
                
                       
        return Items_rel, Items_rel["TOP"]   
    
    
    def comp_imp_unreliability(self):
        
        item_prob = items_unrel(self.time_array, self.item_params)
        
        eqp_list = list(item_prob.keys())
        #P(TOP) with cut sets

        for eqp in eqp_list:
                
            q_cs = []

            for cut_set in self.cut_sets:
                
                p_cs = 1
                        
                for event in cut_set:
                    
                    p_cs *= item_prob[event]
                    
                q_cs.append(p_cs)
                
                
            p_top = 1
            
            for qi in q_cs:
                
                p_top*= (1-qi)
                
            Q_top = 1 - p_top

        #Importance measures calculation

        #P(E|P(eqp)=1) and P(E|P(eqp)=0) calculation

        pe_pq_1 = []
        pe_pq_0 = []


        for eq in eqp_list:
            
            eqp_prob_ = copy.deepcopy(item_prob)

                       
            eqp_prob_[eq]=1
            
            q_cs = []

            for cut_set in self.cut_sets:
                
                p_cs = 1
                        
                for event in cut_set:
                    
                    p_cs *= eqp_prob_[event]
                    
                q_cs.append(p_cs)
               
                
            p_top = 1
            
            for qi in q_cs:
                
                p_top*= (1-qi)
                
            pe_pq_1.append(1 - p_top)
                    

            eqp_prob_[eq]=0
            
            q_cs = []

            for cut_set in self.cut_sets:
                
                p_cs = 1
                        
                for event in cut_set:
                    
                    p_cs *= eqp_prob_[event]
                    
                q_cs.append(p_cs)
                
                
            p_top = 1
            
            for qi in q_cs:
                
                p_top*= (1-qi)
                
            pe_pq_0.append(1 - p_top)
            
            
        comp_imp = pd.DataFrame({'Equipment':eqp_list,
                                 "pe_pq_0":pe_pq_0,
                                 "pe_pq_1":pe_pq_1})

        eqp_prob_ = items_unrel(self.time_array, self.item_params)

        #Importace measures:

        marginal_imp = []
        criticality_imp = []
        diagnostic_imp = []
        raw_imp = []
        rrw_imp = []

        for eq in eqp_list:
            
            marginal_imp.append(comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_1"].values[0]-comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_0"].values[0])
            
            criticality_imp.append((comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_1"].values[0]-comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_0"].values[0])*item_prob[eq]/Q_top)

            diagnostic_imp.append(comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_1"].values[0]*item_prob[eq]/Q_top)

            raw_imp.append(comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_1"].values[0]/Q_top)
            
            rrw_imp.append(Q_top/comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_0"].values[0])


        comp_imp["Marginal"] = marginal_imp         
        comp_imp["Criticality"] = criticality_imp   
        comp_imp["Diagnostic"] = diagnostic_imp   
        comp_imp["RAW"] = raw_imp
        comp_imp["RRW"] = rrw_imp
        comp_imp = comp_imp.drop(['pe_pq_1', 'pe_pq_0'], axis=1)
        
        print(comp_imp)
        
        return (comp_imp)

    def comp_imp_unavailability(self):
        
        item_prob = items_unav(self.time_array, self.item_params)
        
        eqp_list = list(item_prob.keys())
        #P(TOP) with cut sets

        for eqp in eqp_list:
                
            q_cs = []

            for cut_set in self.cut_sets:
                
                p_cs = 1
                        
                for event in cut_set:
                    
                    p_cs *= item_prob[event]
                    
                q_cs.append(p_cs)
                
                
            p_top = 1
            
            for qi in q_cs:
                
                p_top*= (1-qi)
                
            Q_top = 1 - p_top

        #Importance measures calculation

        #P(E|P(eqp)=1) and P(E|P(eqp)=0) calculation

        pe_pq_1 = []
        pe_pq_0 = []


        for eq in eqp_list:
            
            eqp_prob_ = copy.deepcopy(item_prob)

                       
            eqp_prob_[eq]=1
            
            q_cs = []

            for cut_set in self.cut_sets:
                
                p_cs = 1
                        
                for event in cut_set:
                    
                    p_cs *= eqp_prob_[event]
                    
                q_cs.append(p_cs)
               
                
            p_top = 1
            
            for qi in q_cs:
                
                p_top*= (1-qi)
                
            pe_pq_1.append(1 - p_top)
                    

            eqp_prob_[eq]=0
            
            q_cs = []

            for cut_set in self.cut_sets:
                
                p_cs = 1
                        
                for event in cut_set:
                    
                    p_cs *= eqp_prob_[event]
                    
                q_cs.append(p_cs)
                
                
            p_top = 1
            
            for qi in q_cs:
                
                p_top*= (1-qi)
                
            pe_pq_0.append(1 - p_top)
            
            
        comp_imp = pd.DataFrame({'Equipment':eqp_list,
                                 "pe_pq_0":pe_pq_0,
                                 "pe_pq_1":pe_pq_1})

        eqp_prob_ = items_unrel(self.time_array, self.item_params)

        #Importace measures:

        marginal_imp = []
        criticality_imp = []
        diagnostic_imp = []
        raw_imp = []
        rrw_imp = []

        for eq in eqp_list:
            
            marginal_imp.append(comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_1"].values[0]-comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_0"].values[0])
            
            criticality_imp.append((comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_1"].values[0]-comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_0"].values[0])*item_prob[eq]/Q_top)

            diagnostic_imp.append(comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_1"].values[0]*item_prob[eq]/Q_top)

            raw_imp.append(comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_1"].values[0]/Q_top)
            
            rrw_imp.append(Q_top/comp_imp.loc[comp_imp["Equipment"]==eq, "pe_pq_0"].values[0])


        comp_imp["Marginal"] = marginal_imp         
        comp_imp["Criticality"] = criticality_imp   
        comp_imp["Diagnostic"] = diagnostic_imp   
        comp_imp["RAW"] = raw_imp
        comp_imp["RRW"] = rrw_imp
        comp_imp = comp_imp.drop(['pe_pq_1', 'pe_pq_0'], axis=1)
        
        print(comp_imp)
        
        return (comp_imp)




















        



