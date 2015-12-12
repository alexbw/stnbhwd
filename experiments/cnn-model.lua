-- Train and save an MLP on undistorted MNIST
local t = require 'torch'
local grad = require 'autograd'
local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local stn = require 'stn'

return function (batchSize, imageHeight, imageWidth)
   imageHeight = imageHeight or 32
   imageWidth = imageWidth or 32

   -- Set up classifier network
   ---------------------------------
   error("UNIMPLEMENTED")
   model = nn.Sequential()
   model:add(cudnn.SpatialMaxPooling(2,2,2,2))
   model:add(cudnn.SpatialConvolution(1,20,5,5))
   model:add(cudnn.ReLU(true))
   model:add(cudnn.SpatialMaxPooling(2,2,2,2))
   model:add(cudnn.SpatialConvolution(20,20,5,5))
   model:add(cudnn.ReLU(true))
   model:add(nn.View(20*2*2))
   model:add(nn.Linear(20*2*2,20))
   model:add(cudnn.ReLU(true))
   model:add(nn.Linear(20,10))
   model:cuda()

   -- Functionalize networks
   ---------------------------------
   local agClassnet, classParams = grad.functionalize(model)
   local criterion = grad.nn.ClassNLLCriterion()

   -- Set up parameters
   ---------------------------------
   params = {
      classParams = classParams,
   }

   -- Define our loss function
   ---------------------------------
   local function f(inputs, bhwdImages, labels)
      -- Run the classifier on raw images
      local images = torch.view(bhwdImages,
                        batchSize, 
                        imageHeight*imageWidth)

      -- Predict image class on warped image
      local prediction = agClassnet(inputs.classParams, images)

      -- Calculate loss
      local loss = criterion(prediction, labels)

      return loss, prediction, resampledImages
   end

   return f, params
end