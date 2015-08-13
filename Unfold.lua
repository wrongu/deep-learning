--[[ Unfold.lua

A NN module wrapping the torch 'unfold' (and optionally permute) operator
]]--

require 'nn'

local Unfold, parent = torch.class('nn.Unfold', 'nn.Module')

function Unfold:__init(...)
	self.size = {...}
end

function Unfold:outputPermutation(...)
	self.permute = {...}
	self.unpermute = {}
	for fro,to in pairs(self.permute) do
		self.unpermute[to] = fro
	end
end

function Unfold:updateOutput(input)
	self.output = input:unfold(unpack(self.size))
	if self.permute then
		self.output = self.output:permute(unpack(self.permute))
	end
	return self.output
end

function Unfold:updateGradInput(input, gradOutput)
	if self.unpermute then
		self.gradInput = gradOutput:permute(unpack(self.unpermute)):resizeAs(input)
	else
		self.gradInput = gradOutput:resizeAs(input)
	end
	return self.gradInput
end