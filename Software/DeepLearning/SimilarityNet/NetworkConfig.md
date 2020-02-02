
# VanillaNet

## Smaller x2 PS
self.InitNeurons = 26
self.ExpansionFactor = 2.0
self.DropOutRate = 0.7

~ 270fps on BS = 1, Nitin's PC GPU = 0
NumFlops = 197530233
NumParams = 625462
Expected Model Size = 2.391899 MB
warpType = ['pseudosimilarity', 'pseudosimilarity']

## Larger x2 PS
self.InitNeurons = 35
self.ExpansionFactor = 3.0
self.DropOutRate = 0.7
~ 215fps on BS = 1, Nitin's PC GPU = 0
NumFlops = 1023506271
NumParams = 6433776
Expected Model Size = 24.564270 MB
warpType = ['pseudosimilarity', 'pseudosimilarity']

## Larger x4 PS
self.InitNeurons = 24
self.ExpansionFactor = 3.0
self.DropOutRate = 0.7
~ 180fps on BS = 1, Nitin's PC GPU = 0
NumFlops = 986942303
NumParams = 6178188
Expected Model Size = 23.597214 MB
warpType = ['pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity']

## Larger x2 (Scale, Trans)
self.InitNeurons = 35
self.ExpansionFactor = 3.0
self.DropOutRate = 0.7
~ 195fps on BS = 1, Nitin's PC GPU = 0
NumFlops = 1022950605
NumParams = 6294858
Expected Model Size = 24.034340 MB
warpType = ['scale', 'translation']

## Larger x4 (Scale, Scale, Trans, Trans)
self.InitNeurons = 24
self.ExpansionFactor = 3.0
self.DropOutRate = 0.7
~ 170fps on BS = 1, Nitin's PC GPU = 0
NumFlops = 986180244
NumParams = 5987670
Expected Model Size = 22.870445 MB
warpType = ['scale', 'scale', 'translation', 'translation']

Choose best from above. 

# SqueezeNeXt

