image = require 'image'

local sigma = 10
local k = sigma*3
I = image.lena()[1]
h,w = I:size(1),I:size(2)
g = image.gaussian(k,k,sigma,sigma)
local o = torch.conv2(I,g,'F'):narrow(1,k-1,h):narrow(2,k-1,w)
image.display(o)