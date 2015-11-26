local t = require 'torch'
grad = require 'autograd'
require 'nn'
require 'stn'

-- wget 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
-- tar -xf mnist.t7.tgz

require 'cunn'
require 'cudnn'
require 'image'
require 'optim'
paths.dofile('Optim.lua')

batchSize = 5

-- MNIST dataset 
train, validation = require('./mnist.lua')(false)
train.data = train.data:double()
validation.data = validation.data:double()
imgs = train.data:narrow(1,1,batchSize)

-- Reshape for BHWD format
bhwdImages = imgs:transpose(2,3)
bhwdImages = bhwdImages:transpose(3,4)

-- Get the STN functions
-- 
-- Function to generate an affine transform matrix
-- Function to generate a grid
-- Function to generate a bilinear sampler given a grid
-- NOTE: we could probably optimize directly on the grid itself, as opposed to the affine transformation.
fns = grad.functionalize('stn')
matrixGenerator = fns.AffineTransformMatrixGenerator(true,true,true) -- rotation, scale, translation
print(imgs:size())
gridGenerator = fns.AffineGridGeneratorBHWD(imgs:size(3),imgs:size(4))
bilinearSampler = fns.BilinearSamplerBHWD()

-- Generate the transform matrix
rotation = math.pi/4
scale = 1.
translation = {0,0}
transformParams = torch.Tensor{rotation,scale,translation[1],translation[2]}
transformParams = torch.view(transformParams,1,4)
transformParams = torch.expand(transformParams,batchSize,4)
transformMatrix = matrixGenerator(transformParams)

-- Generate the grid
grid = gridGenerator(transformMatrix)

-- Generate the resampled image
resampledImage = bilinearSampler({bhwdImages,grid})

-- Display some stuff
local w1,w2,w3
showImage = resampledImage:select(4,1)
w1=image.display({image=showImage, nrow=16, legend='Just trying to do this', win=w1})

