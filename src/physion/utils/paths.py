import os, sys, importlib.util

def quote(path):
    """ commands are run through a shell: quote paths containing spaces """
    return '"%s"' % path if (' ' in path) else path

# subprocesses (NWB building, pdf summaries, ...) are launched with the
# python running physion (uv / venv / conda environment it was started from)
python_path = quote(sys.executable)

# conda installs where we look for separate environments (suite2p, phy)
possible_conda_dir_lists = [os.path.join(os.path.expanduser('~'), 'miniforge3'),
                            os.path.join(os.path.expanduser('~'), 'miniconda3'),
                            os.path.join(os.path.expanduser('~'), 'anaconda3'),
                            os.path.join(os.path.expanduser('~'), '.conda'),
                            os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Continuum', 'anaconda3'),
                            os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Continuum', 'miniconda3'),
                            os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'miniconda3'),
                            os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'miniforge3')]
                       
def check_path(env='physion'):
    i, success, path = 0, False, python_path
    while (not success) and (i<len(possible_conda_dir_lists)):
        new_path = os.path.join(possible_conda_dir_lists[i], 'envs', env)
        if os.path.isdir(new_path):
            success = True
            if (os.name=='nt'):
                path = quote(os.path.join(new_path, 'python.exe'))
            else:
                path = quote(os.path.join(new_path, 'bin', 'python'))
        i+=1
    return path

# suite2p: the python running physion if suite2p is installed there (e.g. uv environment),
# otherwise a separate "suite2p" conda environment (falls back to python_path if none)
if importlib.util.find_spec('suite2p') is not None:
    python_path_suite2p_env = python_path
else:
    python_path_suite2p_env = check_path('suite2p')

python_path_phy_env = check_path('phy')

LAB = ['Cibele', 'Joana', 'Taddy', 'Sally', 'Sofia', 'Yann']

FOLDERS = {}

for key, val in zip(['~/DATA', 'C:\\DATA', '~/UNPROCESSED', '~/CURATED'],
                    [os.path.join(os.path.expanduser('~'), 'DATA'),
                     "C:\\DATA",
                     os.path.join(os.path.expanduser('~'), 'UNPROCESSED'),
                     os.path.join(os.path.expanduser('~'), 'CURATED')]):
    if os.path.isdir(val):
        FOLDERS[key] = val

for person in LAB:
    for k in ['DATA', 'UNPROCESSED', 'CURATED']:
        key = '~/%s/%s' % (k, person)
        val  = os.path.join(os.path.expanduser('~'), k, person)
        if os.path.isdir(val):
            FOLDERS[key] = val

for key, val in zip(['D:/', 'E:/', 'F:/', 'G:/', 'H:/'],
                    ['D:\\', 'E:\\', 'F:\\', 'G:\\', 'H:\\']):
    if os.path.isdir(val):
        FOLDERS[key] = val
        for person in ['Yann', 'Taddy', 'Joana']:
            if os.path.isdir(os.path.join(val, person)):
                FOLDERS[key+person] = os.path.join(val, person)

    
if __name__=='__main__':
    print('    == folders == ')
    for key in FOLDERS:
        print('  - %s: %s' % (key, FOLDERS[key]))
    print('python_path: ', python_path)
    print('python_path_suite2p_env:', python_path_suite2p_env)

