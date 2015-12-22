-- Train and save an MLP on undistorted MNIST
local t = require 'torch'
local grad = require 'autograd'
local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local stn = require 'stn'

return function (batchSize, imageDepth, imageHeight, imageWidth)

   -- Set up the localizer network
   ---------------------------------
   local locnet = nn.Sequential()
   locnet:add(cudnn.SpatialMaxPooling(2,2,2,2))
   locnet:add(cudnn.SpatialConvolution(imageDepth,20,5,5))
   locnet:add(cudnn.ReLU(true))
   locnet:add(cudnn.SpatialMaxPooling(2,2,2,2))
   locnet:add(cudnn.SpatialConvolution(20,20,5,5))
   locnet:add(cudnn.ReLU(true))
   locnet:add(nn.View(20*2*2))
   locnet:add(nn.Linear(20*2*2,20))
   locnet:add(cudnn.ReLU(true))
   local outLayer = nn.Linear(20,6)
   outLayer.weight:fill(0)
   local bias = torch.FloatTensor(6):fill(0)
   bias[1]=1
   bias[5]=1
   outLayer.bias:copy(bias)
   locnet:add(outLayer)
   locnet:cuda()

   -- Set up classifier network
   ---------------------------------
   model = nn.Sequential()
   model:add(nn.View(imageDepth*imageHeight*imageWidth))
   model:add(nn.Linear(imageDepth*imageHeight*imageWidth, 128))
   model:add(cudnn.ReLU(true))
   model:add(nn.Linear(128, 128))
   model:add(cudnn.ReLU(true))
   model:add(nn.Linear(128, 10))
   model:add(nn.LogSoftMax())
   model:cuda()

   -- Get the STN functions
   ---------------------------------
   local gridGenerator = grad.functionalize(stn.AffineGridGeneratorBHWD(imageHeight, imageWidth)) -- grid is same size as image
   local bilinearSampler = grad.functionalize(stn.BilinearSamplerBHWD())

   -- Functionalize networks
   ---------------------------------
   local agLocnet, locParams = grad.functionalize(locnet)
   local agClassnet, classParams = grad.functionalize(model)
   local criterion = grad.nn.ClassNLLCriterion()

   -- Set up parameters
   ---------------------------------
   params = {
      locParams = locParams,
      classParams = classParams,
   }

   -- Define our loss function
   ---------------------------------
   local function f(inputs, bhwdImages, labels)

      -- Reshape the image for the convnet
      local input = torch.view(bhwdImages,
                        batchSize, imageDepth, imageHeight, imageWidth)

      -- Calculate how we should warp the image
      local warpPrediction = agLocnet(inputs.locParams, torch.contiguous(input))

      -- Get warped grids
      local grids = gridGenerator(torch.view(warpPrediction, batchSize, 2, 3))

      -- Resample the images
      local resampledImages = bilinearSampler({bhwdImages, grids})

      -- Run the classifier on the warped images
      local warpedInput = torch.view(resampledImages,
                        batchSize,
                        imageDepth*imageHeight*imageWidth)

      -- Predict image class on warped image
      local prediction = agClassnet(inputs.classParams, warpedInput)

      -- Calculate loss
      local loss = criterion(prediction, labels)

      return loss, prediction, resampledImages
   end

   return f, params
end