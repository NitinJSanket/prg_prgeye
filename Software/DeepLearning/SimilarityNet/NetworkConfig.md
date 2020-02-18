# Choosing Best ICSTN Warp Architecture: VanillaNet (Model Size <= 25 MB and Model FPS >= 20 FPS on All Cores i7)
- Running on Image Size of 128x128x(3x2)  
- No Data Augmentation on MSCOCO  
- Train on  LR = 1e-4, BatchSize = 32, NumEpochs = 100  

## **BEST ONE!** ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Trans, Trans, Scale, Scale: [100EpochModel](https://drive.google.com/open?id=1aYApkJEegeV6jE0n5MJQr3Ures3bjVeR) 
self.InitNeurons = 18  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 43744396604  
NumParams = 2079870  
Expected Model Size = 23.8192596436 MB  
Network Used: Network.VanillaNet3  
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Scale, Scale, Trans, Trans: [100EpochModel](https://drive.google.com/open?id=1NEQ9gMixBjpzLiUwFsjC6_b7HLP88nm8) 
self.InitNeurons = 18  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 43744396636  
NumParams = 2079870  
Expected Model Size = 23.8192596436 MB  
Network Used: Network.VanillaNet2  
Lambda = [10.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['scale', 'scale', 'translation', 'translation']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Pseudosimilarity x 1: [100EpochModel](https://drive.google.com/open?id=1Pj6Uqr3PeMCJF_vpkd_Nr4ljQ_WfgJiY) 
self.InitNeurons = 36  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 35238910489  
NumParams = 2065935  
Expected Model Size = 23.6512718201 MB  
Network Used: Network.VanillaNet  
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['pseudosimilarity']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Pseudosimilarity x 2: [100EpochModel](https://drive.google.com/open?id=1p4UJ1vybf15NSuWK2m--meqbagxpeR9H)
self.InitNeurons = 26  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 40148554316  
NumParams = 2171890  
Expected Model Size = 24.8676147461 MB  
Network Used: Network.VanillaNet  
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['pseudosimilarity', 'pseudosimilarity']    

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Pseudosimilarity x 4: [100EpochModel](https://drive.google.com/open?id=1IhJLQ0rc1mPV6ZrQkFiBFK8Nhkp8hMjl) 
self.InitNeurons = 18  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 43749760630  
NumParams = 2107524  
Expected Model Size = 24.1357345581 MB  
Network Used: Network.VanillaNet  
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Scale, Trans [100EpochModel](https://drive.google.com/open?id=1fw3vzSNM0VSy8vz6wXbPqd5l0tbmB_E3) 
self.InitNeurons = 26  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 40144680527  
NumParams = 2151919  
Expected Model Size = 24.6390647888 MB  
Network Used: Network.VanillaNet2Simpler  
Lambda = [10.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['scale', 'translation']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) Trans, Scale [100EpochModel](https://drive.google.com/open?id=1PnYX1PXEgZsQ6UUbYeD9uS3OxDzXVX4Y) 
self.InitNeurons = 26  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops =   
NumParams =   
Expected Model Size =  MB  
Network Used: Network.VanillaNet3Simpler  
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
warpType = ['translation', 'scale']  


**Best Combination is 2T2S.**  

# Choosing Best Large Network Architecture (Model Size <= 25 MB and Model FPS >= 20 FPS on All Cores i7)
- Running on Image Size of 128x128x(3x2)  
- No Data Augmentation on MSCOCO  
- Train on  LR = 1e-4, BatchSize = 32, NumEpochs = 100  

## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) ResNet [50EpochModel](https://drive.google.com/open?id=1CVcOdikijaZzUGBUeTwCtMZJ7k2M8Az5) [100EpochModel](https://drive.google.com/open?id=1q2vSRg2_LSkkEkL9X46Lz4TVkKlSrQbY) 
self.InitNeurons = 13  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 55175035536  
NumParams = 2119578  
Expected Model Size = 24.268951416 MB  
Network Used: Network.ResNet3  
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  


## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) ResNet [50EpochModel](https://drive.google.com/open?id=1jT7kfmUtdMdLisf7851-t1umaKSSDBPv) 
self.InitNeurons = 13  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 55175035536  
NumParams = 2119578  
Expected Model Size = 24.268951416 MB  
Network Used: Network.ResNet3  
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  

## ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) SqueezeNet [50EpochModel](https://drive.google.com/open?id=1psmiRJwUj_iZ_2qA-wRGmjVrlXEBPmFz) [100EpochModel](https://drive.google.com/open?id=1tHkZ8YW6I3jdolJQRZ1XyWE4kKE7vXTz) 
self.InitNeurons = 12    
self.ExpansionFactor = 1.2 
self.DropOutRate = 0.7  
self.NumBlocks = 2  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 209774502264  
NumParams = 2120962  
Expected Model Size = 24.2732849121 MB  
Network Used: Network.SqueezeNet3    
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  

## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) MobileNetv1 [50EpochModel]() [100EpochModel]() Currently Training 50 Epoch Model on Nitin's GPU 0 
self.InitNeurons = 14   
self.ExpansionFactor = 1.95 
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 37762557976  
NumParams = 2041798  
Expected Model Size = 23.389084 MB  
Network Used: Network.MobileNetv13    
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  oscillates a lot
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation] trains at LR = 1e-5
warpType = ['translation', 'translation', 'scale', 'scale']  


## ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) ShuffleNetv2 [50EpochModel]() [100EpochModel]() Currently Training 50 Epoch Model on Nitin's GPU 1 
self.InitNeurons = 16  
self.ExpansionFactor = 2.0 
self.DropOutRate = 0.7  
self.NumBlocks = 3  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 111691692124  
NumParams = 2103110  
Expected Model Size = 24.1038360596 MB  
Network Used: Network.ShuffleNetv23    
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation] 
warpType = ['translation', 'translation', 'scale', 'scale']  

# Choosing Best Small Network Architecture (Model Size <= 2.5 MB and Model FPS >= 200 FPS on All Cores i7)
- Running on Image Size of 128x128x(3x2)  
- No Data Augmentation on MSCOCO  
- Train on  LR = 1e-4, BatchSize = 32, NumEpochs = 100  

## ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) VanillaNet: [50EpochModel](https://drive.google.com/open?id=1aYApkJEegeV6jE0n5MJQr3Ures3bjVeR) 
self.InitNeurons = 10  
self.ExpansionFactor = 2.0  
self.DropOutRate = 0.7  
self.NumBlocks = 2  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 16597160988  
NumParams = 208286  
Expected Model Size = 2.38822937012 MB  
Network Used: Network.VanillaNet3Small  
Lambda = [1.0, 10.0, 10.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  

## ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) ResNet [50EpochModel]() 
self.InitNeurons = 8 
self.ExpansionFactor = 1.95 
self.DropOutRate = 0.7  
self.NumBlocks = 2  
?? FPS on BS = 1, Nitin's PC All Cores i7  
NumFlops = 18004062284  
NumParams = 195466  
Expected Model Size = 2.24032592773MB  
Network Used: Network.ResNet3Small    
Lambda = [1.0, 1.0, 1.0] # [Scale, Translation]  
warpType = ['translation', 'translation', 'scale', 'scale']  


- ![#f03c15](https://placehold.it/15/f03c15/000000?text=+) `#f03c15` Red  
- ![#c5f015](https://placehold.it/15/c5f015/000000?text=+) `#c5f015` Green   
- ![#1589F0](https://placehold.it/15/1589F0/000000?text=+) `#1589F0` Blue  


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
