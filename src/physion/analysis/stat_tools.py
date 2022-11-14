import os, sys, itertools, pathlib
import numpy as np
from scipy import stats

class StatTest:
    
    def __init__(self, x, y,
                 test='wilcoxon',
                 positive=False,
                 verbose=True):

        self.x, self.y = np.array(x), np.array(y)
        for key in ['pvalue', 'statistic']:
            setattr(self, key, 1)
        self.positive = positive # to evaluate positive only deflections

        try:
            self.r = stats.pearsonr(x, y)[0] # Pearson's correlation coef
            self.sign = np.mean(y-x)>0 # sign of the effect

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
        except (ValueError, TypeError):
            if verbose:
                print(' -----------------   ')
                print('x, y = ', x, y)
                print('  statistical test failed   ')
                print(' -----------------   ')
            self.r, self.sign = 0, 0
            self.pvalue, self.statistic = 1, 0


    def significant(self, threshold=0.01):
        """
        here with 
        """

        if (self.pvalue is not None) and (self.pvalue<threshold):
            if self.positive and not self.sign:        
                return False
            else:
                return True
        elif (self.pvalue is not None):
            return False
        else:
            print(' /!\ no valid p-value for significance test !! /!\ ')
            return False

    def pval_annot(self, size=5):
        """
        uses the 
        """
        if self.positive and not self.sign:        
            return 'n.s.', size
        elif self.pvalue<1e-3:
            return '***', size+1
        elif self.pvalue<1e-2:
            return '**', size+1
        elif self.pvalue<0.05:
            return '*', size+1
        else:
            return 'n.s.', size
        

        
if __name__=='__main__':


    # filename = os.path.join(os.path.expanduser('~'), 'DATA', 'CaImaging', 'Wild_Type_GCamp6f', '2021_03_23-11-26-36.nwb')
    
    #filename = sys.argv[-1]
    #FullData= Data(filename)
        
    StatTest(None, None)
