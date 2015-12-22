local torch = require 'torch'
local g = require 'autograd'
local stn = g.functionalize('stn')
local image = require 'image'

local function distortImage(image, scale, rotation, translationX, translationY)
   scale = scale or 1
   rotation = rotation or 0
   translationX = translationX or 0
   translationY = translationY or 0

   local imageDepth, imageHeight, imageWidth
   if image:nDimension() == 2 then
      imageHeight, imageWidth = image:size(1), image:size(2)
   elseif image:nDimension() == 3 then
      imageDepth, imageHeight, imageWidth = image:size(1), image:size(2), image:size(3)
   else
      error("Unsupported dimension number " .. image:nDimension())
   end

   local gridGenerator = stn.AffineGridGeneratorBHWD(imageHeight, imageWidth)
   local matrixGenerator = stn.AffineTransformMatrixGenerator(true,true,true)
   local resampler = stn.BilinearSamplerBHWD()

   local params = torch.FloatTensor{{rotation, scale, translationX, translationY}}
   local A = matrixGenerator(params)
   local I = torch.transpose(torch.view(image, imageDepth or 1, imageHeight, imageWidth, 1), 1, 4):contiguous()
   local grid = gridGenerator(A)
   local out = resampler({I, gridGenerator(A)})

   if imageDepth then
      return torch.transpose(torch.transpose(torch.squeeze(out, 1),2,3),1,2), torch.squeeze(A)
   else
      return torch.squeeze(torch.squeeze(out, 1), 3), torch.squeeze(A)
   end
end

function distortData(images)
   local stn = g.functionalize('stn')
   local nImages = images:size(1)
   local transforms = torch.zeros(nImages, 2, 3):float()
   for i=1,nImages do
      if i % 1000 == 0 then
         print(i)
      end
      -- Sample rotation, translation, scale
      image = images:select(1,i)
      local scale = torch.uniform(0.7, 1.2)
      local rotation = torch.uniform(-3.14/4, 3.14/4)
      local translationX = torch.uniform(-0.5, 0.5)
      local translationY = torch.uniform(-0.5, 0.5)
      local I, A = distortImage(image, scale, rotation, translationX, translationY)
      transforms[i] = A
      image:copy(I)
   end
   return images, transforms
end


local testFileName = 'mnist.t7/test_32x32.t7'
local trainFileName = 'mnist.t7/train_32x32.t7'

local train = torch.load(trainFileName, 'ascii')
local test = torch.load(testFileName, 'ascii')
train.data = train.data:float()
train.labels = train.labels:float()
test.data = test.data:float()
test.labels = test.labels:float()

torch.manualSeed(0)

train.data, train.transform = distortData(train.data)
test.data, test.transform = distortData(test.data)

torch.save('mnist.t7/test_32x32-distort.t7', test, 'ascii')
torch.save('mnist.t7/train_32x32-distort.t7', train, 'ascii')
