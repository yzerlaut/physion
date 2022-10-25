import os

python_path = 'python'

possible_conda_dir_lists = [os.path.join(os.path.expanduser('~'), 'miniconda3'),
                            os.path.join(os.path.expanduser('~'), 'anaconda3'),
                            os.path.join(os.path.expanduser('~'), '.conda'),
                            os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Continuum', 'anaconda3'),
                            os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Continuum', 'miniconda3')]
                       
def check_path(env='physion'):
    i, success, path = 0, False, python_path
    while (not success) and (i<len(possible_conda_dir_lists)):
        new_path = os.path.join(possible_conda_dir_lists[i], 'envs', env)
        if os.path.isdir(new_path):
            success = True
            if (os.name=='nt'):
                path = os.path.join(new_path, 'python.exe')
            else:
                path = os.path.join(new_path, 'bin', 'python')
        i+=1
    return path

if (os.name=='nt') and os.path.isdir(os.path.join(os.path.expanduser('~'), '.conda', 'envs', 'acquisition')):
    print('acq setting')
    python_path = os.path.join(os.path.expanduser('~'), '.conda', 'envs', 'acquisition', 'python.exe')
else:
    python_path = check_path('physion')

        
python_path_suite2p_env = check_path('suite2p')


FOLDERS = {}
for key, val in zip(['~/DATA', '~/UNPROCESSED', '~/CURATED'],
                    [os.path.join(os.path.expanduser('~'), 'DATA'),
                     os.path.join(os.path.expanduser('~'), 'UNPROCESSED'),
                     os.path.join(os.path.expanduser('~'), 'CURATED')]):
    if os.path.isdir(val):
        FOLDERS[key] = val

for key, val in zip(['D-drive', 'E-drive', 'F-drive', 'G-drive', 'H-drive'],
                    ['D:\\', 'E:\\', 'F:\\', 'G:\\', 'H:\\']):
    if os.path.isdir(val):
        FOLDERS[key] = val

for user in ['yann', 'yann.zerlaut']:
    for key, val in zip(['storage-curated', 
                         'storage-data',
                         'usb (YANN)',
                         'usb (Yann)',
                         'usb (code)'],
                        ['/media/%s/DATADRIVE1/CURATED/' % user,
                         '/media/%s/DATADRIVE1/DATA/' % user,
                         '/media/%s/YANN/' % user,
                         '/media/%s/Yann/' % user,
                         '/media/%s/CODE_YANN/']):
        if os.path.isdir(val):
            FOLDERS[key] = val
        
FOLDERS['10.0.0.1:curated'] = 'yann@10.0.0.1:/media/yann/DATADRIVE1/CURATED'
# FOLDERS['MsWin-data'] = '/media/yann/Windows/home/yann/DATA/'
# FOLDERS['MsWin-cygwin'] = '/media/yann/Windows/Users/yann.zerlaut/DATA/'
FOLDERS['10.0.0.1:~/DATA'] = 'yann@10.0.0.1:/home/yann/DATA/'
FOLDERS['10.0.0.2:~/DATA'] = 'yann@10.0.0.2:/home/yann/DATA/'


    
if __name__=='__main__':
    print('    == folders == ')
    for key in FOLDERS:
        print('  - %s: %s' % (key, FOLDERS[key]))
    print('python_path: ', python_path)
    print('python_path_suite2p_env:', python_path_suite2p_env)

