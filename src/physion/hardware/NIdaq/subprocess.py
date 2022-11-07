#from subprocess import Popen, PIPE, STDOUT

# p = Popen(['grep', 'f'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)

# grep_stdout = p.communicate(input='one\ntwo\nthree\nfour\nfive\nsix\n')[0]
# print grep_stdout

def NIdaq_rec(self, duration, filename, dt=1e-3):
    p = Popen("python hardware_control\\NIdaq\\recording.py -T %.2f -dt %.5f -f %s" % (duration, dt, filename))
    return p


if __name__=='__main__':

    pass

