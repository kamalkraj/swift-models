// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import TensorFlow

import Python

let np = Python.import("numpy")
let image = Python.import("matplotlib.image")
let cm = Python.import("matplotlib.cm")

/// Reads a file into an array of bytes.
func readFile(_ path: String) -> [UInt8] {
    let possibleFolders  = [".", "GAN"]
    for folder in possibleFolders {
        let parent = URL(fileURLWithPath: folder)
        let filePath = parent.appendingPathComponent(path)
        guard FileManager.default.fileExists(atPath: filePath.path) else {
            continue
        }
        let data = try! Data(contentsOf: filePath, options: [])
        return [UInt8](data)
    }
    print("File not found: \(path)")
    exit(-1)
}

/// Reads MNIST images and labels from specified file paths.
func readMNIST(imagesFile: String, labelsFile: String) -> (images: Tensor<Float>,
                                                           labels: Tensor<Int32>) {
    print("Reading data.")
    let images = readFile(imagesFile).dropFirst(16).map(Float.init)
    let labels = readFile(labelsFile).dropFirst(8).map(Int32.init)
    let rowCount = labels.count

    print("Constructing data tensors.")
    return (
        images: Tensor(shape: [rowCount, 784], scalars: images)/255,
        labels: Tensor(labels)
    )
}

/// A DiscriminatorNet.
struct DiscriminatorNet: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var hidden1a = Dense<Float>(inputSize: 784, outputSize: 1024, activation: relu)
    var dropout1a = Dropout<Float>(probability: 0.3)
    
    var hidden1b = Dense<Float>(inputSize: 1024, outputSize: 512, activation: relu)
    var dropout1b = Dropout<Float>(probability: 0.3)

    var hidden1c = Dense<Float>(inputSize: 512, outputSize: 256, activation: relu)
    var dropout1c = Dropout<Float>(probability: 0.3)

    var out = Dense<Float>(inputSize: 256, outputSize: 1, activation: sigmoid)

    @differentiable
    func call(_ input: Input) -> Output {
        let hiddenFeatures = input.sequenced(through:hidden1a, dropout1a, hidden1b, dropout1b)
        return hiddenFeatures.sequenced(through: hidden1c, dropout1c, out)
    }
}

struct GeneratorNet: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var hidden1a = Dense<Float>(inputSize: 100, outputSize: 256, activation: relu)
    
    var hidden1b = Dense<Float>(inputSize: 256, outputSize: 512, activation: relu)

    var hidden1c = Dense<Float>(inputSize: 512, outputSize: 1024, activation: relu)
    
    var out = Dense<Float>(inputSize: 1024, outputSize: 784, activation: tanh)

    @differentiable
    func call(_ input: Input) -> Output {
        let hiddenFeatures = input.sequenced(through:hidden1a, hidden1b, hidden1c, out)
        return hiddenFeatures
    }
}

let epochCount = 200
let batchSize = 512

func minibatch<Scalar>(in x: Tensor<Scalar>, at index: Int) -> Tensor<Scalar> {
    let start = index * batchSize
    return x[start..<start+batchSize]
}

let (trainImages, _) = readMNIST(imagesFile: "train-images-idx3-ubyte",
                                        labelsFile: "train-labels-idx1-ubyte")

var discriminator = DiscriminatorNet()
var generator = GeneratorNet()

let discriminatorOptimizer = Adam(for: discriminator, learningRate: 0.0002)
let generatorOptimizer = Adam(for: generator, learningRate: 0.0002)

print("Beginning training...")

func noise(size: Int) -> Tensor<Float>{
    let randomNoise = Tensor<Float>(randomUniform: [size,100])
    return randomNoise
}

func realDataTarget(size: Int) -> Tensor<Float> {
    let data = Tensor<Float>(ones: [size, 1])
    return data
}

func fakeDataTarget(size: Int) -> Tensor<Float> {
    let data = Tensor<Float>(zeros: [size, 1])
    return data
}

let numTestSamples = 16
let testNoise = noise(size: numTestSamples)



// The training loop.
for epoch in 1...epochCount {
    var discriminatorLoss: Float = 0
    var generatorLoss: Float = 0
    Context.local.learningPhase = .training
    for i in 0 ..< Int(trainImages.shape[0]) / batchSize {
        let realData = minibatch(in: trainImages, at: i)
        // Generate fake data
        let fakeData = generator(noise(size: realData.shape[0]))
        // Train Discriminator.
        let 𝛁discriminatorModel = discriminator.gradient { discriminator -> Tensor<Float> in
            // Train on real data
            let predictionReal = discriminator(realData)
            let realLoss = sigmoidCrossEntropy(logits: predictionReal, labels: realDataTarget(size: realData.shape[0]))
            // Train on fake data
            let predictionFake = discriminator(fakeData)
            let fakeLoss = sigmoidCrossEntropy(logits: predictionFake, labels: fakeDataTarget(size: realData.shape[0]))
            // Combine loss
            let loss = realLoss + fakeLoss
            discriminatorLoss += loss.scalarized()
            return loss
        }
        // Update the discriminator model's differentiable variables along the gradient vector.
        discriminatorOptimizer.update(&discriminator.allDifferentiableVariables, along: 𝛁discriminatorModel)

        // Train Generator
        let 𝛁generatorModel = generator.gradient { generator -> Tensor<Float> in
            // Generate fake data
            let fakeData = generator(noise(size: realData.shape[0]))
            // Predict generated fake data using discriminator
            let prediction = discriminator(fakeData)
            // Calcualte loss
            let loss = sigmoidCrossEntropy(logits: prediction, labels: realDataTarget(size: realData.shape[0]))

            generatorLoss += loss.scalarized()

            return loss
        }
        generatorOptimizer.update(&generator.allDifferentiableVariables,along: 𝛁generatorModel)
    
    }
    print("""
          [Epoch \(epoch)] \
          Discriminator Loss: \(discriminatorLoss), \
          Generator Loss: \(generatorLoss)
          """)
    Context.local.learningPhase = .inference
    let testImages = generator(testNoise).reshaped(to: [numTestSamples,28,28])
    let testImagesNumpy: PythonObject = testImages.makeNumpyArray()
    let fileName: String = "images/" + String(epoch) + "_" 
    for i in 0..<numTestSamples {
        let file = fileName + String(i) + ".png"
        image.imsave(file, testImagesNumpy[i], Python.None, Python.None, cm.binary)
    }
}
