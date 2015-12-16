-- wget 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
-- tar -xf mnist.t7.tgz



function distortData32(foo)
   local res=torch.FloatTensor(foo:size(1), 1, 32, 32):fill(0)
   local distImg=torch.FloatTensor(1, 42, 42):fill(0)
   for i=1,foo:size(1) do
      baseImg=foo:select(1,i)
     
      r = image.rotate(baseImg, 0) -- torch.uniform(-3.14/4,3.14/4))
      scale = torch.uniform(0.7,1.2)
      sz = torch.floor(scale*32)
      s = image.scale(r, sz, sz)
      rest = 42-sz
      offsetx = torch.random(1, 1+rest)
      offsety = torch.random(1, 1+rest)
      
      distImg:zero()
      distImg:narrow(2, offsety, sz):narrow(3,offsetx, sz):copy(s)
      res:select(1,i):copy(image.scale(distImg, 32, 32))
   end
   return res
end

function createDatasetsDistorted(batch, normalize, distort, batchSize)
   local batch = batch or false
   if normalize ~= false then
      local normalize = normalize or true
   end
   if distort ~= false then
      local distort = distort or true
   end
   local batchSize = batchSize or 256

   local testFileName = 'mnist.t7/test_32x32.t7'
   local trainFileName = 'mnist.t7/train_32x32.t7'

   local train = torch.load(trainFileName, 'ascii')
   local test = torch.load(testFileName, 'ascii')
   train.data = train.data:float()
   train.labels = train.labels:float()
   test.data = test.data:float()
   test.labels = test.labels:float()
   
   -- distortion 
   if distort then  
      train.data = distortData32(train.data)
      test.data = distortData32(test.data)
   end

   -- normalization
   if normalize then
      local mean = train.data:mean()
      local std = train.data:std()
      train.data:add(-mean):div(std)
      test.data:add(-mean):div(std)
   end
   
   
   if batch then
      local datasetTrain = {
         getBatch = function(self, idx)
            local data = train.data:narrow(1, (idx - 1) * batchSize + 1, batchSize)
            local labels = train.labels:narrow(1, (idx - 1) * batchSize + 1, batchSize)
            return data, labels, batchSize
         end,
         getNumBatches = function()
            return torch.floor(60000 / batchSize)
         end,
         data = train.data,
         labels = train.labels,
      }
      
      local datasetVal = {
         getBatch = function(self, idx)
            local data = test.data:narrow(1, (idx - 1) * batchSize + 1, batchSize)
            local labels = test.labels:narrow(1, (idx - 1) * batchSize + 1, batchSize)
            return data, labels, batchSize
         end,
         getNumBatches = function()
            return torch.floor(10000 / batchSize)
         end,
         data = test.data,
         labels = test.labels
      }
      return datasetTrain, datasetVal
   else
      return train, test
   end
end

return createDatasetsDistorted