--[[ Maxout.lua

written by Richard Lange

Implements a Maxout nn Layer from 
	"Maxout Networks" Ian J. Goodfellow, David Warde-Farley, 
	Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013

This implementation simply concatenates Convolution with MaxPooling inside a Sequential() Module

A Maxout layer creates nOutputPlane*nMaxOver intermediate features that are MAXed over the nMaxOver
	dimension, ending with simply nOutputPlane features. That is, nOutputPlane corresponds to 'm' and
	nMaxOver to 'k' in the Goodfellow et al. paper.

input parameters with the 'c' suffix refer to the convolution step, and 'm' corresponds to the max pooling step

input --[convolution]--> Z --[max pooling]--> output

input: nInputPlane x iWidth x iHeight

Z: nOutputPlane x nMaxOver x zWidth x zHeight; where
   zWidth  = (iWidth  + 2*padWc - kWc) / dWc + 1
   zHeight = (iHeight + 2*padHc - kHc) / dHc + 1

output: nOutputPlane x oWidth x oHeight; where
   oWidth  = (zWidth + 2*padWm - kWm) / dWm + 1
   oHeight = (zHeigh + 2*padHm - kHm) / dHm + 1

]]--

require 'nn'
require 'Unfold'

local Maxout, parent = torch.class('nn.Maxout', 'nn.Sequential')

function Maxout:__init(nInputPlane, nOutputPlane, nMaxOver, kWc, kHc, dWc, dHc, padWc, padHc, kWm, kHm, dWm, dHm)
	parent.__init(self)

	local convolution_layer = nn.SpatialConvolution(
		nInputPlane,
		nOutputPlane * nMaxOver, -- will be expanded to 'time' dimension in reshape to 4D
		kWc, kHc, dWc, dHc, padWc, padHc)

	-- Unfold to expand the first 'nOutputPlane*nMaxOver' dimension into two, so we get a 4d
	-- tensor of size nOutputPlane x width x height x nMaxOver
	local reshape3d_to_4d = nn.Unfold(1, nMaxOver, nMaxOver)
	-- permute to make nMaxOver dimension the second one
	reshape3d_to_4d:outputPermutation(1,4,2,3)

	-- max over the 'time' dimension, i.e. the 2nd dimension in the expansion above
	local max_pooling_layer = nn.VolumetricMaxPooling(nMaxOver, kWm, kHm, 1, dWm, dHm)

	-- collapse the (now singleton) 'time' dimension
	local reshape4d_to_3d = nn.Select(2,1) -- analogous to output:squeeze(2)

	self:add(convolution_layer)
	self:add(reshape3d_to_4d)
	self:add(max_pooling_layer)
	self:add(reshape4d_to_3d)
end

function Maxout:updateParameters(learningRate)
	parent.updateParameters(self, learningRate)
	if self.kernelNormLimit then
		local convLayer = self:get(1)
		-- note that SpatialConvolution has a viewWeights() method that views as nOut x (nIn*kW*kH),
		-- but we want to view as (nOut*nIn) x (kW*kH) to isolate each kernel as a row
		local view = convLayer.weight:view(convLayer.weight:size(1)*convLayer.weight:size(2), -1)
		-- ensure the L2-norm of each kernel does not exceed kernelNormLimit
		-- (modifying view modifies convLayer.weight as well)
		view:renorm(view, 2, 1, self.kernelNormLimit) -- '2' is L2 norm, '1' is slices along the first dimension
	end
end

function Maxout:limitKernelNorm(limit)
	self.kernelNormLimit = limit
	return self
end

function Maxout:reset(stdv)
	self:get(1):reset(stdv)
end