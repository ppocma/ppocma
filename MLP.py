import numpy as np
import matplotlib.pyplot as pp
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf

useUnitNormInit=True

class Layer:
    def __init__(self,input:tf.Tensor,initInput:tf.Tensor,nUnits:int,useSkips=True,activation=None):
        X=input
        xDim=X.shape[1].value

        #Default forward pass ops
        #Initial weight matrix rows are random and unit-norm.
        #in other words, if input is unit-variance, output variables are also unit-variance 
        if useUnitNormInit:
            initialW=np.random.normal(0,1,size=[nUnits,xDim])
            initialW=initialW/np.linalg.norm(initialW,axis=1,keepdims=True)
        else:
            #MSRA initialization (https://www.tensorflow.org/api_docs/python/tf/contrib/layers/variance_scaling_initializer)
            initialW=tf.truncated_normal([nUnits,xDim], 0.0, stddev=tf.sqrt(2.0 / xDim))
        self.W=tf.Variable(initial_value=initialW,dtype=tf.float32,name='W')
        self.b=tf.Variable(initial_value=np.zeros([1,nUnits]),dtype=tf.float32,name='b')
        h=tf.matmul(X,tf.transpose(self.W))+self.b
        if activation=="relu":
            h=tf.nn.relu(h)
        elif activation=="selu":
            h=tf.nn.selu(h)
        elif activation=="lrelu":
            h=tf.nn.leaky_relu(h,alpha=0.1)
        elif activation=="tanh":
            h=tf.nn.tanh(h)
        elif activation=="swish":
            h=tf.nn.swish(h)
        elif activation is not None:
            raise NameError("Invalid activation type ({}) for Layer".format(activation))
        if useSkips:
            self.output=tf.concat([X,h],axis=1)
        else:
            self.output=h

        #Init ops for data-dependent initialization pass.
        #Same functionality as above, but we init biases such that the input distribution's mean maps to zero output.
        X=initInput
        xMean=tf.reduce_mean(X,axis=0,keepdims=True)
        b=tf.assign(self.b,-tf.matmul(xMean,tf.transpose(self.W)))
        h=tf.matmul(X,tf.transpose(self.W))+b
        if activation=="relu":
            h=tf.nn.relu(h)
        elif activation=="selu":
            h=tf.nn.selu(h)
        elif activation=="lrelu":
            h=tf.nn.leaky_relu(h,alpha=0.1)
        elif activation=="tanh":
            h=tf.nn.tanh(h)
        elif activation=="swish":
            h=tf.nn.swish(h)
        elif activation is not None:
            raise NameError("Invalid activation type ({}) for Layer".format(activation))
        if useSkips:
            self.initOutput=tf.concat([X,h],axis=1)
        else:
            self.initOutput=h


        
class MLP:
    def __init__(self,input:tf.Tensor,nLayers:int,nUnitsPerLayer:int, nOutputUnits:int, activation="lrelu", firstLinearLayerUnits:int=0, useSkips:bool=True):
        self.layers=[]
        X=input
        initX=input

        #add optional first linear layer (useful, e.g., for reducing the dimensionality of high-dimensional outputs
        #which reduces the parameter count of all subsequent layers)
        if firstLinearLayerUnits!=0:
            layer=Layer(X,initX,firstLinearLayerUnits,useSkips=False,activation=None)
            self.layers.append(layer)
            X,initX=layer.output,layer.initOutput

        #add hidden layers
        for layerIdx in range(nLayers):
            layer=Layer(X,initX,nUnitsPerLayer,useSkips=useSkips,activation=activation)
            self.layers.append(layer)
            X,initX=layer.output,layer.initOutput

        #add output layer
        layer=Layer(X,initX,nOutputUnits,useSkips=False,activation=None)
        self.layers.append(layer)
        self.output,self.initOutput=layer.output,layer.initOutput



    #This method returns a list of assign ops that can be used with a sess.run() call to copy
    #all network parameters from a source network. This is useful, e.g., for implementing a slowly updated
    #target network in Reinforcement Learning
    def copyFromOps(self,src):
        result=[]
        for layerIdx in range(len(self.layers)):
            result.append(tf.assign(self.layers[layerIdx].W,src.layers[layerIdx].W))
            result.append(tf.assign(self.layers[layerIdx].b,src.layers[layerIdx].b))
        return result

#functional interface
def mlp(input:tf.Tensor,nLayers:int,nUnitsPerLayer:int, nOutputUnits:int, activation="selu", firstLinearLayerUnits:int=0,useSkips:bool=True):
    instance=MLP(input,nLayers,nUnitsPerLayer,nOutputUnits,activation,firstLinearLayerUnits)
    return instance.output,instance.initOutput

#simple test: 
if __name__ == "__main__":
    print("Generating toy data")
    x=[]
    y=[]
    maxAngle=5*np.pi
    discontinuousTest=True
    if discontinuousTest:
        maxAngle=np.pi
        for angle in np.arange(0,maxAngle,0.01):
            x.append(angle)
            if angle>maxAngle*0.8:
                y.append(0.0)
            else:
                y.append(np.sin(angle)*np.sign(np.sin(angle*10)))
    else:
        for angle in np.arange(0,maxAngle,0.1):
            r=angle*0.15
            x.append(angle)
            if angle>maxAngle*0.8:
                y.append(0.0)
            else:
                y.append(r*np.sin(angle))

    x=np.array(x)
    y=np.array(y)
    x=np.reshape(x,[x.shape[0],1])
    y=np.reshape(y,[y.shape[0],1])
    interpRange=0.2
    xtest=np.arange(-interpRange+np.min(x),np.max(x)+interpRange,0.001)
    xtest=np.reshape(xtest,[xtest.shape[0],1])
    
    print("Initializing matplotlib")
    pp.figure(1)
    pp.subplot(3,1,1)
    pp.scatter(x[:,0],y[:,0])
    pp.ylabel("data")

    print("Creating model")
    sess=tf.InteractiveSession()
    tfX=tf.placeholder(dtype=tf.float32,shape=[None,1])
    tfY=tf.placeholder(dtype=tf.float32,shape=[None,1])
    #IMPORTANT: deep networks benefit immensely from data-dependent initialization.
    #This is why the constructor returns the initial predictions separately - to initialize, fetch this tensor in a sess.run with 
    #the first minibatch. See the sess.run below
    predictions,initialPredictions=mlp(input=tfX,nLayers=8,nUnitsPerLayer=8,nOutputUnits=1,activation="lrelu")
    optimizer=tf.train.AdamOptimizer()
    loss=tf.losses.mean_squared_error(tfY,predictions)
    optimize=optimizer.minimize(loss)
  
    print("Initializing model")
    tf.global_variables_initializer().run(session=sess)
    #This sess.run() initializes the network biases based on x, and also returns the initial predictions.
    #It is noteworthy that with this initialization, even a deep network has zero-mean output with variance similar to input.
    networkOut=sess.run(initialPredictions,feed_dict={tfX:x})
    pp.subplot(3,1,2)
    pp.scatter(x[:,0],networkOut[:,0])
    pp.ylabel("initialization")
    pp.draw()
    pp.pause(0.001)


    print("Optimizing")
    nIter=8000
    for iter in range(nIter):
        temp,currLoss=sess.run([optimize,loss],feed_dict={tfX:x,tfY:y})
        if iter % 100 == 0:
            print("Iter {}/{}, loss {}".format(iter,nIter,currLoss))
    networkOut=sess.run(predictions,feed_dict={tfX:x})
    pp.subplot(3,1,3)
    pp.scatter(x[:,0],networkOut[:,0])
    pp.ylabel("trained")

    pp.show()