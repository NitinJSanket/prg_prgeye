# Choosing Best ICSTN Warp Architecture: VanillaNet (Model Size <= 25 MB and Model FPS >= 20 FPS on All Cores i7)
- Running on Image Size of 128x128x(3x2)
- No Data Augmentation on MSCOCO
- Train on  LR = 1e-3, BatchSize = 32, NumEpochs = 100

## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Trans, Trans, Scale, Scale: [Trained Model Link]() Currently Training on PRGUMD's GPU 0
self.InitNeurons = 18
self.ExpansionFactor = 2.0
self.DropOutRate = 0.7
self.NumBlocks = 3
?? FPS on BS = 1, Nitin's PC All Cores i7
NumFlops = 43744396604
NumParams = 2079870
Expected Model Size = 23.8192596436 MB
Network Used: Network.VanillaNet3
warpType = ['translation', 'translation', 'scale', 'scale']

## ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) Scale, Scale, Trans, Trans: [Trained Model Link]()
self.InitNeurons = 18
self.ExpansionFactor = 2.0
self.DropOutRate = 0.7
self.NumBlocks = 3
?? FPS on BS = 1, Nitin's PC All Cores i7
NumFlops = 43744396636
NumParams = 2079870
Expected Model Size = 23.8192596436 MB
Network Used: Network.VanillaNet2
warpType = ['scale', 'scale', 'translation', 'translation']

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Pseudosimilarity x 1: [Trained Model Link](https://drive.google.com/open?id=1Pj6Uqr3PeMCJF_vpkd_Nr4ljQ_WfgJiY) 
self.InitNeurons = 36
self.ExpansionFactor = 2.0
self.DropOutRate = 0.7
self.NumBlocks = 3
?? FPS on BS = 1, Nitin's PC All Cores i7
NumFlops = 35238910489
NumParams = 2065935
Expected Model Size = 23.6512718201 MB
Network Used: Network.VanillaNet
warpType = ['pseudosimilarity']

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Pseudosimilarity x 2: [Trained Model Link](https://drive.google.com/open?id=1p4UJ1vybf15NSuWK2m--meqbagxpeR9H)
self.InitNeurons = 26
self.ExpansionFactor = 2.0
self.DropOutRate = 0.7
self.NumBlocks = 3
?? FPS on BS = 1, Nitin's PC All Cores i7
NumFlops = 40148554316
NumParams = 2171890
Expected Model Size = 24.8676147461 MB
Network Used: Network.VanillaNet
warpType = ['pseudosimilarity', 'pseudosimilarity']

## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) Pseudosimilarity x 4: [Trained Model Link]() Currently Training on Nitin's GPU 1
self.InitNeurons = 18
self.ExpansionFactor = 2.0
self.DropOutRate = 0.7
self.NumBlocks = 3
?? FPS on BS = 1, Nitin's PC All Cores i7
NumFlops = 43749760630
NumParams = 2107524
Expected Model Size = 24.1357345581 MB
Network Used: Network.VanillaNet
warpType = ['pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity']


**Maybe** If PS x 4 outperforms PS x 2 and PS x 1

## ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) Scale, Trans
self.InitNeurons = 45
self.ExpansionFactor = 2.0
self.DropOutRate = 0.7
self.NumBlocks = 3
~ 162.15fps on BS = 1, Nitin's PC All Cores i7
NumFlops = 1098047365
NumParams = 6395493
Expected Model Size = 24.418156 MB
Network Used: Network.VanillaNet
warpType = ['scale', 'translation']

## ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) Trans, Scale
self.InitNeurons = 45
self.ExpansionFactor = 2.0
self.DropOutRate = 0.7
self.NumBlocks = 3
~ 162.15fps on BS = 1, Nitin's PC All Cores i7
NumFlops = 1098047365
NumParams = 6395493
Expected Model Size = 24.418156 MB
Network Used: Network.VanillaNet
warpType = ['scale', 'translation']


<!-- 

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
self.InitNeurons = 12
self.ExpansionFactor = 1.9
self.DropOutRate = 0.7
self.NumBlocks = 3
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
~ 174.03fps on BS = 1, Nitin's PC CPU All Cores -->
