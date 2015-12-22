-- Train and save an MLP on undistorted MNIST
local t = require 'torch'
local grad = require 'autograd'
local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local stn = require 'stn'

return function (batchSize, imageDepth, imageHeight, imageWidth)
   imageDepth = imageDepth or 1
   imageHeight = imageHeight or 32
   imageWidth = imageWidth or 32

   -- Set up classifier network
   ---------------------------------
   local model = nn.Sequential()
   model:add(nn.View(imageDepth*imageHeight*imageWidth))
   model:add(nn.Linear(imageDepth*imageHeight*imageWidth, 128))
   model:add(cudnn.ReLU(true))
   model:add(nn.Linear(128, 128))
   model:add(cudnn.ReLU(true))
   model:add(nn.Linear(128, 10))
   model:add(nn.LogSoftMax())
   model:cuda()

   -- Functionalize networks
   ---------------------------------
   local agClassnet, classParams = grad.functionalize(model)
   local criterion = grad.nn.ClassNLLCriterion()

   -- Set up parameters
   ---------------------------------
   local params = {
      classParams = classParams,
   }

   -- Define our loss function
   ---------------------------------
   local function f(inputs, bhwdImages, labels)
      -- Run the classifier on raw images
      local images = torch.view(bhwdImages,
                        batchSize,
                        imageDepth,
                        imageHeight*imageWidth)

      -- Predict image class
      local prediction = agClassnet(inputs.classParams, images)

      -- Calculate loss
      local loss = criterion(prediction, labels)

      return loss, prediction, resampledImages
   end

   return f, params
end