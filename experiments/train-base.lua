-- Train and save an MLP on undistorted MNIST
require 'trepl'
local t = require 'torch'
local grad = require 'autograd'
local nn = require 'nn'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local optim = require 'optim'
local image = require 'image'
local stn = require 'stn'
local hasQT,q = pcall(require, 'qt')

-- Options
local opt = lapp [[
Run benchmarks.

Options:
   --distort (default false)
   --model (default stn)
   --optimizer (default adagrad)
   --display (default false)
   --batchSize (default 256)
   --nEpoch (default 100)
]]
if opt.distort == "true" then opt.distort = true else opt.distort = false end
if opt.display == "true" then opt.display = true else opt.display = false end

torch.manualSeed(0)

-- Name of file to serialize once fitting is completed
outFile = string.format("model=%s-distort=%s-optimizer=%s-nepoch=%d-batchSize=%d.t7", 
   opt.model, 
   tostring(opt.distort), 
   opt.optimizer, 
   opt.nEpoch,
   opt.batchSize)


-- MNIST dataset 
---------------------------------
local train, validation = paths.dofile("../demo/distort_mnist.lua")(true, true, opt.distort, opt.batchSize) -- batch, normalize, distort
local imageHeight, imageWidth = train.data:size(3), train.data:size(4)
local gridHeight = imageHeight
local gridWidth = imageWidth

-- Set up confusion matrix
---------------------------------
local confusionMatrix = optim.ConfusionMatrix({0,1,2,3,4,5,6,7,8,9})

-- Initialize the model
---------------------------------
if opt.model == "stn" then
   f, params = paths.dofile('stn-model.lua')(opt.batchSize, imageWidth, imageHeight)
elseif opt.model == "mlp" then
   f, params = paths.dofile('mlp-model.lua')(opt.batchSize, imageWidth, imageHeight)
else
   print("Unrecognized model " .. opt.model)
end

-- Get the gradient of the model
--------------------------------- 
local g = grad(f, {optimize = true})

-- Set up the optimizer
--------------------------------- 
local state, states, optimfn
if opt.optimizer == "adagrad" then
   state = {learningRate=1e-2}
   optimfn, states = grad.optim.adagrad(g, state, params)
elseif opt.optimizer == "adam" then
   local state = {learningRate = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8}
   local optimfn, states = grad.optim.adam(g, state, params)
else
   print("Unrecognized optimizer " .. opt.optimizer)
end

local w1, w2 -- for display
local nBatches = train:getNumBatches()

for epoch=1,opt.nEpoch do
   print("Epoch " .. epoch)
   for i=1,nBatches do
      xlua.progress(i,nBatches)

      -- Get images in BHWD format, labels in one-hot format:
      local data, labels, n = train:getBatch(i)
      local bhwdImages = data:transpose(2,3):cuda():transpose(3,4):cuda()
      local target = labels:cuda()

      -- Calculate gradients:
      local grads, loss, prediction, resampledImages = optimfn(bhwdImages, target)

      -- Log performance:
      confusionMatrix:batchAdd(prediction, target)
      if i % 50 == 0 then
         print(confusionMatrix)
         confusionMatrix:zero()
         if hasQT and opt.display and resampledImages then
            local transformedImage = resampledImages:select(4,1)
            local origImage = bhwdImages:select(4,1)
            w1=image.display({image=origImage, nrow=16, legend='Original', win=w1})
            w2=image.display({image=transformedImage, nrow=16, legend='Resampled', win=w2})
         end
      end
   end
end


torch.save(outFile, {f=f, params=params, fns=fns})