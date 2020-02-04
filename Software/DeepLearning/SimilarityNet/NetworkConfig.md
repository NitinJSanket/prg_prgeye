
# VanillaNet (Model Size <= 2.5 MB and <=25 MB)

## Smaller x2 PS
self.InitNeurons = 26
self.ExpansionFactor = 2.0
self.DropOutRate = 0.7
self.NumBlocks = 3

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

# SqueezeNet 
Speed calculated as best of 1000 runs
## Smaller (Model FPS >= 200 FPS on All Cores i7, and Model Size <= 2.5 MB)
self.InitNeurons = 4
self.ExpansionFactor = 1.42
self.DropOutRate = 0.7
warpType = ['pseudosimilarity', 'pseudosimilarity']
NumFlops = 65174577
NumParams = 401246
Expected Model Size = 1.530754 MB
~ 199.07fps on BS = 1, Nitin's PC CPU All Cores

## Larger (Model FPS >= 20 FPS on All Cores i7, and Model Size <= 25 MB)
self.InitNeurons = 16
self.ExpansionFactor = 1.5
self.DropOutRate = 0.7
warpType = ['pseudosimilarity', 'pseudosimilarity']
NumFlops = 7799809633
NumParams = 6082914
Expected Model Size = 23.205086 MB
~ 31.70fps on BS = 1, Nitin's PC CPU All Cores


# ResNet 
Speed calculated as best of 1000 runs
## Smaller (Model FPS >= 200 FPS on All Cores i7, and Model Size <= 2.5 MB)
NumRes = 4
self.InitNeurons = 15
self.ExpansionFactor = 1.5
self.DropOutRate = 0.7
warpType = ['pseudosimilarity', 'pseudosimilarity']
NumFlops = 161618397
NumParams = 611100
Expected Model Size = 2.337128 MB
~ 223.63fps on BS = 1, Nitin's PC CPU All Cores

## Larger (Model FPS >= 20 FPS on All Cores i7, and Model Size <= 25 MB)
NumRes = 4
self.InitNeurons = 16
self.ExpansionFactor = 2
self.DropOutRate = 0.7
warpType = ['pseudosimilarity', 'pseudosimilarity']
NumFlops = 485211389
NumParams = 6317446
Expected Model Size = 24.114525 MB
~ 174.03fps on BS = 1, Nitin's PC CPU All Cores