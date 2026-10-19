import os
import numpy as np


def markdown_table(dataset,
                   keys=['files', 'viruses', 'genotypes',
                         'subjects', 'ages', 'protocol'],
                   append_to=None):
    """
    prints (and returns) a markdown table of a dataset
        (the output of "scan_folder_for_NWBfiles")

    the 'files' column shows markdown links to the "filename.md" notes
    rows sorted by genotype, then virus, then age

    with "append_to=path", the "## Dataset" title and the table are
    appended to that (existing) markdown or text file
    """

    def age_key(age):
        # non-numeric ages (e.g. None) at the end
        try:
            return (0, float(age))
        except (TypeError, ValueError):
            return (1, 0.)

    isorted = sorted(range(len(dataset['files'])),
                     key=lambda i: (str(dataset['genotypes'][i]),
                                    str(dataset['viruses'][i]),
                                    age_key(dataset['ages'][i])))

    def cell(key, i):
        if key=='files':
            filename = os.path.basename(dataset['files'][i]).replace('.nwb', '')
            return '[%s](%s.md)' % (filename, filename)
        return str(dataset[key][i]).replace('|', '\\|')

    lines = ['| ' + ' | '.join(keys) + ' |',
             '|' + '|'.join([' --- ' for key in keys]) + '|']
    for i in isorted:
        lines.append('| ' + ' | '.join([cell(key, i) for key in keys]) + ' |')

    table = '\n'.join(lines)
    print(table)

    if append_to is not None:
        if not os.path.isfile(append_to):
            raise FileNotFoundError('"%s" is not an existing file' % append_to)
        with open(append_to) as f:
            content = f.read()
        # start on a new line, separated by an empty line
        sep = '' if content=='' else\
                ('\n' if content.endswith('\n') else '\n\n')
        with open(append_to, 'a') as f:
            f.write(sep + '## Dataset\n\n' + table + '\n')

    return table
