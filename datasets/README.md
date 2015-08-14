### Datasets

This directory contains modules for loading datasets. Each dataset provides `['train']`, `['validate']`, and `['test']` sets.

Example usage (mnist):

	data = require('datasets/mnist')
	traindata = data['train']

	for idx=1,#traindata do
		local example = traindata[idx]
		local image = example[1]
		local class = example[1]
	end