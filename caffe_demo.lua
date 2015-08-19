-- caffe demo

require 'caffe'
require 'image'

net = caffe.Net(paths.concat(os.getenv('CAFFE_DIR'), 'models/bvlc_alexnet/deploy.prototxt'), 'models/caffe-binaries/bvlc_alexnet.caffemodel', 'test')
img = image.load('datasets/binaries/suitguy.jpg'):resize(3,227,227):float()

input = torch.FloatTensor(10,3,227,227)
for dim=1,10 do
	input:narrow(1,dim,1):copy(img)
end

out = net:forward(input)
print(out)