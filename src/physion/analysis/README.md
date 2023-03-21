# Analysis

## load 

```
filename = ''

import physion
data = physion.analysis.read_NWB(filename)

episodes = physion.analysis.process_NWB.EpisodeData(data,
                                                    protocol_name='whatever-protocol-name')
```
