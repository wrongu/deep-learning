--[[
mnist_mlp.lua

written by Richard Lange

A simple MLP; classic sigmoid activation 4-layer neural net
]]--

require 'nn'

mlp = nn.Sequential()
-- 1x28x28 input image
mlp:add(nn.Reshape(28*28))
-- 784x1
mlp:add(nn.Linear(784, 300))
mlp:add(nn.Sigmoid())
-- 300x1
mlp:add(nn.Linear(300, 100))
mlp:add(nn.Sigmoid())
-- 100x1
mlp:add(nn.Linear(100,10))
-- 10x1 output classes (note: class 1 is 0, class 10 is 9.. careful off off by one)
mlp:add(nn.LogSoftMax())

return mlp