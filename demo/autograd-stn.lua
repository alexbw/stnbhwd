local t = require 'torch'
local grad = require 'autograd'
local stn = require 'stn'
local image = require 'image'

local batchSize = 256

-- MNIST dataset 
local train, validation = require('./distort_mnist.lua')(false, false, false) -- batch, normalize, distort
train.data:div(255.0)
train.data = train.data:cuda()
validation.data = validation.data:cuda()
local imgs = train.data:narrow(1,1,batchSize)

-- Reshape for BHWD format
local bhwdImages = imgs:transpose(2,3)
local bhwdImages = bhwdImages:transpose(3,4)

-- Get the STN functions
---------------------------------
-- * Function to generate an affine transform matrix
-- * Function to generate a grid
-- * Function to generate a bilinear sampler given a grid
-- NOTE: we could probably optimize directly on the grid itself, as opposed to the affine transformation.
local fns = grad.functionalize('stn')
local matrixGenerator = fns.AffineTransformMatrixGenerator(true,true,true) -- rotation, scale, translation
local gridGenerator = fns.AffineGridGeneratorBHWD(imgs:size(3),imgs:size(4))
local bilinearSampler = fns.BilinearSamplerBHWD()


-- Define our silly function (rewards zooming in on bright pixels, we hope)
local function f(inputs, bhwdImages)
   local M = matrixGenerator(inputs.transformParams)
   local grids = gridGenerator(M)
   local resampledImage = bilinearSampler({bhwdImages, grids})
   return -torch.sum(resampledImage), resampledImage
end

-- Generate the transform matrix
local rotation = 0
local scale = 1
local translation = {0,0}
local params = torch.Tensor{rotation,scale,translation[1],translation[2]}
params = torch.view(params,1,4)
params = torch.expand(params,batchSize,4):cuda()

-- Take the gradient of the function
local g = grad(f, {optimize = true})

local w1
local lr = 1e-5
for i=1,1000 do
   local grads, loss, resampledImages = g({transformParams=params}, bhwdImages)

   local showImage = resampledImages:select(4,1)
   w1=image.display({image=showImage, nrow=16, min=0, max=1, scaleeach=false, saturate=false, legend='Resampled', win=w1})

   -- Update
   local update = grads.transformParams*lr

   -- Optionally dampen the learning rate
   -- lr = lr * (1-1e-4)

   -- Update the params
   params = params - update

end

