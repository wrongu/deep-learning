--[[
mnist.lua

written by Richard Lange

provides MNIST training and testing data, based on torch tutorial from University of Toronto

Each example is {[1x28x28], [10x1]}, where 
	example[1] is a FloatTensor/image with pixels in [0,1]
	example[2] is an indicator vector of the correct class (nine 0s, one 1)
]]--

require '../utils'

-- note that 'paths' is module of torch
local root = paths.dirname(paths.thisfile())

local test_file = paths.concat(root, 'binaries', 'mnist-train-images.t7')
if not paths.filep(test_file) then
	error("MNIST binary files are missing; expecting " .. test_file .. ", a 60000x28x28 Tensor")
end

local dataset = {}

dataset.train_images = torch.load(paths.concat(root, 'binaries', 'mnist-train-images.t7')):split(1,1)
dataset.train_labels = torch.load(paths.concat(root, 'binaries', 'mnist-train-labels.t7')):split(1,1)
dataset.test_images = torch.load(paths.concat(root, 'binaries', 'mnist-test-images.t7')):split(1,1)
dataset.test_labels = torch.load(paths.concat(root, 'binaries', 'mnist-test-labels.t7')):split(1,1)

-- first 50k examples are training
dataset['train'] = zip_tables(dataset.train_images, dataset.train_labels, 1, 50000)
-- 50001 to 60000 examples are the dev set
dataset['validate'] = zip_tables(dataset.train_images, dataset.train_labels, 50001, 60000)
-- the 10k test examples are just that
dataset['test'] = zip_tables(dataset.test_images, dataset.test_labels, 1, 10000)

return dataset