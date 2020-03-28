# VanillaNet
## Float32 Large
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/VanillaNet/Float32 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Float32 --InitNeurons=36 --NetworkName=Network.VanillaNet --ModelPrefix=VanillaNet --GPUDevice=-1 

## Int8 Large
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/VanillaNet/Int8 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Int8 --InitNeurons=36 --NetworkName=Network.VanillaNet --ModelPrefix=VanillaNet --GPUDevice=-1 

## Float32 Small
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/VanillaNet/Float32 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Float32 --InitNeurons=20 --NetworkName=Network.VanillaNetSmall --ModelPrefix=VanillaNetSmall --GPUDevice=-1 

## Int8 Small
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/VanillaNet/Int8 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Int8 --InitNeurons=20 --NetworkName=Network.VanillaNetSmall --ModelPrefix=VanillaNetSmall --GPUDevice=-1 

# ResNet
## Float32 Large
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/ResNet/Float32 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Float32 --InitNeurons=26 --NetworkName=Network.ResNet --ModelPrefix=ResNet --GPUDevice=-1 

## Int8 Large
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/ResNet/Int8 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Int8 --InitNeurons=26 --NetworkName=Network.ResNet --ModelPrefix=ResNet --GPUDevice=-1 

## Float32 Small
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/ResNet/Float32 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Float32 --InitNeurons=16 --NetworkName=Network.ResNetSmall --ModelPrefix=ResNetSmall --GPUDevice=-1 

## Int8 Small
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/ResNet/Int8 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Int8 --InitNeurons=16 --NetworkName=Network.ResNetSmall --ModelPrefix=ResNetSmall --GPUDevice=-1 

# SqueezeNet
## Float32 Large
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/SqueezeNet/Float32 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Float32 --InitNeurons=24 --NetworkName=Network.SqueezeNet --ModelPrefix=SqueezeNet --GPUDevice=-1 

## Int8 Large
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/SqueezeNet/Int8 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Int8 --InitNeurons=24 --NetworkName=Network.SqueezeNet --ModelPrefix=SqueezeNet --GPUDevice=-1 

## Float32 Small
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/SqueezeNet/Float32 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Float32 --InitNeurons=20 --NetworkName=Network.SqueezeNetSmall --ModelPrefix=SqueezeNetSmall --GPUDevice=-1 

## Int8 Small
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/SqueezeNet/Int8 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Int8 --InitNeurons=20 --NetworkName=Network.SqueezeNetSmall --ModelPrefix=SqueezeNetSmall --GPUDevice=-1 

# MobileNetv1
## Float32 Large
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/MobileNetv1/Float32 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Float32 --InitNeurons=25 --NetworkName=Network.MobileNetv1 --ModelPrefix=MobileNetv1 --GPUDevice=-1 

## Int8 Large
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/MobileNetv1/Int8 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Int8 --InitNeurons=25 --NetworkName=Network.MobileNetv1 --ModelPrefix=MobileNetv1 --GPUDevice=-1 

## Float32 Small
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/MobileNetv1/Float32 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Float32 --InitNeurons=16 --NetworkName=Network.MobileNetv1Small --ModelPrefix=MobileNetv1Small --GPUDevice=-1 

## Int8 Small
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/MobileNetv1/Int8 --Mode=Train --TFLite=1 --EdgeTPU=1 --TFLiteOpt=Latency --TFLiteQuant=Int8 --InitNeurons=16 --NetworkName=Network.MobileNetv1Small --ModelPrefix=MobileNetv1Small --GPUDevice=-1 

# ShuffleNetv2
** DOES NOT CONVERT TO TFLITE**
## Float32 Large
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/ShuffleNetv2/Float32 --Mode=Train --InitNeurons=32 --NetworkName=Network.ShuffleNetv2 --ModelPrefix=ShuffleNetv2 --GPUDevice=-1 

## Float32 Small
./CreateNetwork.py --CheckPointPath /mnt/d/Git/SpeedTests/ShuffleNetv2/Float32 --Mode=Train --InitNeurons=14 --NetworkName=Network.ShuffleNetv2Small --ModelPrefix=ShuffleNetv2Small --GPUDevice=-1 

