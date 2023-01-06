import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


class BestDistribution:
    
    def __init__(self, time_array):
        
        self.time_array = time_array
       
        
    def get_best_distribution(self):
        
        params = {"norm":st.norm.fit(self.time_array),
                 "weibull_min":st.weibull_min.fit(self.time_array, floc=0),
                 "expon":st.expon.fit(self.time_array, floc = 0),
                 "lognorm":st.lognorm.fit(self.time_array)}
        
        dist_results = []

        for param in params:
            
            # Applying the Kolmogorov-Smirnov test
            D, p = st.kstest(self.time_array, param, args=params[param])
            
            dist_results.append((param, p))

        # select the best fitted distribution
        best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
        # store the name of the best fit and its p value

        if best_dist == "weibull_min":
            
            dist = "Weibull"
            shape_parameter = params[best_dist][0]
            scale_parameter = params[best_dist][2]
            
            return dist, (shape_parameter, scale_parameter)
        
        if best_dist == "norm":
            
            dist = "Normal"
            shape_parameter = params[best_dist][0]
            scale_parameter = params[best_dist][1]

            return dist, (shape_parameter, scale_parameter)
        
        if best_dist == "expon":
            
            dist = "Exponential"
            scale_parameter = params[best_dist][1]
       
            return dist, (scale_parameter)
        
        if best_dist == "lognorm":
            
            dist = "Lognormal"
            shape_parameter = params[best_dist][0]
            scale_parameter = params[best_dist][1]
            
            return dist, (shape_parameter, scale_parameter)
        
# Failure/repair rate

def weibull_rf(t, param):
    
    return st.weibull_min.pdf(t, param[0], scale = param[1])/(1 - st.weibull_min.cdf(t, param[0], scale = param[1]))

def norm_rf(t, param):
    
    return st.norm.pdf(t, param[0], param[1])/(1-st.norm.cdf(t, param[0], param[1]))

def expon_rf(t, param):
    
    return st.expon.pdf(t,  scale = param[0])/(1-st.expon.cdf(t, scale = param[0]))

def lognorm_rf(t, param):
    
    return st.lognorm.pdf(t, param[0], scale = param[1])/(1-st.lognorm.cdf(t, param[0], scale = param[1]))
    

class AnalysisPlot:
    
    def __init__(self, ttf, ttr, comp):        
      
        self.ttf = ttf
        self.ttr = ttr
        self.comp = comp
        
        
    def reliability(self, distribution, dist_parameters):
        
        time_array = np.arange(0, self.ttf.max()*1.02, round(self.ttf.max()/50))
        
        if distribution == "Weibull":
            
            pdf_f = st.weibull_min.pdf(time_array, 
                                       dist_parameters[0], 
                                       scale = dist_parameters[1])
            
            rel_f = 1-st.weibull_min.cdf(time_array, 
                                         dist_parameters[0], 
                                         scale = dist_parameters[1])
            
            fr_f = weibull_rf(time_array, dist_parameters)
            
            p = "shape_par = {} - scale_par = {}".format(round(dist_parameters[0], 3),
                                                                round(dist_parameters[1], 3))


        elif distribution == "Normal":
            
            pdf_f = st.norm.pdf(time_array, dist_parameters[0], dist_parameters[1])
            
            rel_f = 1-st.norm.cdf(time_array, dist_parameters[0], dist_parameters[1])
            
            fr_f = norm_rf(time_array, dist_parameters)

            p = "shape_par = {} - nscale_par = {}\n\n".format(round(dist_parameters[0], 3),
                                                                round(dist_parameters[1], 3))    

        elif distribution == "Exponential":
            
            pdf_f = st.norm.pdf(time_array,  scale = dist_parameters[0])
            
            rel_f = 1-st.norm.cdf(time_array, scale =dist_parameters[0])
            
            fr_f = expon_rf(time_array, dist_parameters)
            
            p = "lamba = {} \n\n".format(round(1/self.dist_parameters[0], 3))
            
        elif distribution == "Lognormal":
            
            pdf_f = st.lognorm.pdf(time_array,  scale = dist_parameters[0])
            
            rel_f = 1-st.lognorm.cdf(time_array, scale = dist_parameters[0])
            
            fr_f = lognorm_rf(time_array, dist_parameters)

            p = "shape_par = {} \n\nscale_par = {}\n\n".format(round(dist_parameters[0], 3),
                                                                round(dist_parameters[0], 3))
            
            
        item_analysis, ((ttf_hist, pdf_fun), (rel_av_graph, failure_rate)) = plt.subplots(2,2)
        item_analysis.suptitle('{} - RELIABILITY ANALYSIS'.format(self.comp), fontsize=30)
        item_analysis.set_figheight(15)
        item_analysis.set_figwidth(25)
        item_analysis.subplots_adjust(wspace=.25)


        # ttf distribution graph
        ttf_hist.set_title('TTF DISTRIBUTION', fontsize=20)
        ttf_hist.set_ylabel('[N°]')
        ttf_hist.set_xlabel('Time [h]')
        ttf_hist.grid(visible=True, which='major', zorder=3)
        item_analysis.text(0.1,0.925,
                      'DISTRIBUTION = {} - '.format(distribution)+
                      p +
                      '- MTBF = {} [h] - '.format(round(self.ttf.mean(), 1)) +
                      ' STD = {} [h] - '.format(round(self.ttf.std(), 1)),
                      fontsize = 15,
                      ha = 'left', va = "center")

        counts, bins = np.histogram(self.ttf, bins=np.arange(0,self.ttf.max()+50,50))
        ttf_hist.hist(bins[:-1], bins, weights=counts)
            
            
        # pdf graph
        pdf_fun.set_title('PROBABILITY DISTRIBUTION FUNCTION', fontsize=20)
        #pdf_fun.set_ylim(ymax=max(st.weibull_min.pdf(times, par[0], scale = par[2])))
        pdf_fun.set_ylabel('[%]')
        pdf_fun.set_xlabel('Time [h]')
        pdf_fun.grid(visible=True, which='major', zorder=3)


        pdf_fun.plot(time_array, pdf_f)    


        # reliability/availability function

        rel_av_graph.set_title('RELIABILITY (t)', fontsize=20)

        rel_av_graph.set_ylabel('[%]')
        rel_av_graph.set_xlabel('Time [h]')
        rel_av_graph.grid(visible=True, which='major', zorder=3)

        rel_av_graph.plot(time_array, rel_f)    
           

        # failure rate function

        failure_rate.set_title('FAILURE RATE (t)', fontsize=20)
        #pdf_fun.set_ylim(ymax=max(st.weibull_min.pdf(times, par[0], scale = par[2])))
        failure_rate.set_ylabel('[%]')
        failure_rate.set_xlabel('Time [h]')
        failure_rate.grid(visible=True, which='major', zorder=3)

        failure_rate.plot(time_array, fr_f)   
        plt.show()
        
    def maintainability(self, distribution, dist_parameters):
        
        time_array = np.arange(0, self.ttr.max()*1.02, round(self.ttr.max()/50, 5))
        
        if distribution == "Weibull":
            
            pdf_f = st.weibull_min.pdf(time_array, 
                                       dist_parameters[0], 
                                       scale = dist_parameters[1])
            
            main_f = st.weibull_min.cdf(time_array, 
                                         dist_parameters[0], 
                                         scale = dist_parameters[1])
            
            rr_f = weibull_rf(time_array, dist_parameters)
            
            p = "shape_par = {} - scale_par = {}".format(round(dist_parameters[0], 3),
                                                                round(dist_parameters[1], 3))


        elif distribution == "Normal":
            
            pdf_f = st.norm.pdf(time_array, dist_parameters[0], dist_parameters[1])
            
            main_f = st.norm.cdf(time_array, dist_parameters[0], dist_parameters[1])
            
            rr_f = norm_rf(time_array, dist_parameters)

            p = "shape_par = {} - nscale_par = {} -".format(round(dist_parameters[0], 3),
                                                                round(dist_parameters[1], 3))    

        elif distribution == "Exponential":
            
            pdf_f = st.norm.pdf(time_array,  scale = dist_parameters[0])
            
            main_f = st.norm.cdf(time_array, scale =dist_parameters[0])
            
            rr_f = expon_rf(time_array, dist_parameters)
            
            p = "lamba = {} - ".format(round(1/dist_parameters[0], 3))
            
        elif distribution == "Lognormal":
            
            pdf_f = st.lognorm.pdf(time_array,  scale = dist_parameters[0])
            
            main_f = st.lognorm.cdf(time_array, scale = dist_parameters[0])
            
            rr_f = lognorm_rf(time_array, dist_parameters)

            p = "shape_par = {} - scale_par = {} - ".format(round(dist_parameters[0], 3),
                                                                round(dist_parameters[0], 3))
            
            
        item_analysis, ((ttr_hist, pdf_fun), (main_graph, repair_rate)) = plt.subplots(2,2)
        item_analysis.suptitle('{} - MAINTAINABILITY ANALYSIS'.format(self.comp), fontsize=30)
        item_analysis.set_figheight(15)
        item_analysis.set_figwidth(25)
        item_analysis.subplots_adjust(wspace=.25)


        # ttf distribution graph
        ttr_hist.set_title('TTR DISTRIBUTION', fontsize=20)
        ttr_hist.set_ylabel('[N°]')
        ttr_hist.set_xlabel('Time [h]')
        ttr_hist.grid(visible=True, which='major', zorder=3)
        item_analysis.text(0.1,0.925,
                      'DISTRIBUTION = {} - '.format(distribution)+
                      p +
                      '- MTBF = {} [h] - '.format(round(self.ttr.mean(), 1)) +
                      ' STD = {} [h]'.format(round(self.ttr.std(), 1)),
                      fontsize = 15,
                      ha = 'left', va = "center")

        counts, bins = np.histogram(self.ttr, bins=np.arange(0,self.ttr.max()+0.5,0.5))
        ttr_hist.hist(bins[:-1], bins, weights=counts)
            
            
        # pdf graph
        pdf_fun.set_title('PROBABILITY DISTRIBUTION FUNCTION', fontsize=20)
        #pdf_fun.set_ylim(ymax=max(st.weibull_min.pdf(times, par[0], scale = par[2])))
        pdf_fun.set_ylabel('[%]')
        pdf_fun.set_xlabel('Time [h]')
        pdf_fun.grid(visible=True, which='major', zorder=3)


        pdf_fun.plot(time_array, pdf_f)    


        # reliability/availability function

        main_graph.set_title(' MAINTAINABILITY (t)', fontsize=20)

        main_graph.set_ylabel('[%]')
        main_graph.set_xlabel('Time [h]')
        main_graph.grid(visible=True, which='major', zorder=3)

        main_graph.plot(time_array, main_f)    
           

        # failure rate function

        repair_rate.set_title('REPAIR RATE (t)', fontsize=20)
        #pdf_fun.set_ylim(ymax=max(st.weibull_min.pdf(times, par[0], scale = par[2])))
        repair_rate.set_ylabel('[%]')
        repair_rate.set_xlabel('Time [h]')
        repair_rate.grid(visible=True, which='major', zorder=3)

        repair_rate.plot(time_array, rr_f)   
        plt.show()
        
    def availability(self):
                          
        mu = 1/self.ttr.mean()
        lam = 1/self.ttf.mean()
        
        times = np.arange(0, self.ttf.max()*1.1, 1)

        plt.figure(figsize=(15,10))
        plt.text(0.1,0.905, 
                 'MTBF = {} [h]\n\n'.format(round(self.ttf.mean(),1))+
                 'MTTR = {} [h]\n\n'.format(round(self.ttr.mean(),1))+
                 "AV_inf = {} [h]".format(round(self.ttf.mean()/(self.ttf.mean()+self.ttr.mean()), 3)),
                 fontsize = 20)
        plt.ylabel('[%]')
        plt.xlabel('Time [h]')
        plt.title('{} - AVAILABILITY (t)'.format(self.comp), fontsize=25)
        plt.ylim(0.9,1.001)
        plt.grid(visible=True, which='major', zorder=3)
        plt.plot(times, (mu/(lam+mu) + (lam/(lam+mu))*(np.exp(-(lam+mu)*times))))
        plt.show()

        







