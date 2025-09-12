import os, sys, itertools, pathlib
import numpy as np
from scipy import stats

class StatTest:
    
    def __init__(self, x, y,
                 test='wilcoxon',
                 sign='both', # 
                 positive=False, # DEPRECATED as of 09/2025 - use "sign" instead
                 verbose=True):
        """
        statistical test object

        sign can be either: "both" (default), "positive", "negative"
        
        """

        if positive:
            self.sign = 'positive'
            print(' "positive" arg is deprecated, switch to sign="positive" ')
        else:
            self.sign = sign

        self.x, self.y = np.array(x), np.array(y)
        for key in ['pvalue', 'statistic']:
            setattr(self, key, 1)

        try:
            self.r = stats.pearsonr(x, y)[0] # Pearson's correlation coef

            if test=='wilcoxon':
                result = stats.wilcoxon(self.x, self.y)
                for key in ['pvalue', 'statistic']:
                    setattr(self, key, getattr(result, key))
                self.statistic = result.statistic
            elif test=='anova':
                result = stats.f_oneway(self.x, self.y)
                for key in ['pvalue', 'statistic']:
                    setattr(self, key, getattr(result, key))
            elif test=='ttest':
                result = stats.ttest_rel(self.x, self.y)
                for key in ['pvalue', 'statistic']:
                    setattr(self, key, getattr(result, key))
            else:
                print(' "%s" test not implemented ! ' % test)

        except BaseException as be:
            print(' -----------------   ')
            print(be)
            print('x, y = ', x, y)
            print('  statistical test failed   ')
            print(' -----------------   ')
            self.r = 0
            self.pvalue, self.statistic = 1, 0


    def significant(self, threshold=0.01):
        """
        here with 
        """

        if (self.pvalue is not None) and (self.pvalue<threshold):
            if self.sign=='positive' and np.mean(self.y-self.x)<=0:
                return False
            elif self.sign=='negative' and np.mean(self.y-self.x)>=0:
                return False
            else:
                return True
        elif (self.pvalue is not None):
            return False
        else:
            print(' [!!] no valid p-value for significance test !! [!!] ')
            return False

    def pval_annot(self, size=5):
        """
        uses the following annotation rule:
        - n.s. : p>=0.05
        - * : 0.01<p<=0.05
        - ** : 0.001<p<=0.01
        - *** : p<0.001
        """
        if not self.significant(threshold=0.05):
            return 'n.s.', size
        elif self.pvalue<1e-3:
            return '***', size+1
        elif self.pvalue<1e-2:
            return '**', size+1
        elif self.pvalue<0.05:
            return '*', size+1
        else:
            print(' stat_tools --> this should never happen !')
            return 'n.s.', size
        

        
if __name__=='__main__':


    filename = sys.argv[-1]
    FullData= Data(filename)
        
    StatTest(None, None)
