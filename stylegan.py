from PIL import Image
from math import floor, log2
import numpy as np
import time
from functools import partial
from random import random
import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import tensorflow.keras.backend as K

from datagen import dataGenerator, printProgressBar

im_size = 256
latent_size = 512
BATCH_SIZE = 12
directory = "Earth"

cha = 48

n_layers = int(log2(im_size) - 1)

mixed_prob = 0.9

def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size]).astype('float32')

def noiseList(n):
    return [noise(n)] * n_layers

def mixedList(n):
    tt = int(random() * n_layers)
    p1 = [noise(n)] * tt
    p2 = [noise(n)] * (n_layers - tt)
    return p1 + [] + p2

def nImage(n):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1]).astype('float32')


#Loss functions
def gradient_penalty(samples, output, weights):
    gradients = K.gradients(output, samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    # (weight / 2) * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weights)

def hinge_d(y_true, y_pred):
    return K.mean(K.relu(1.0 + (y_true * y_pred)))

def w_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


#Lambdas
def AdaIN(x):
    #Normalize x[0]
    mean = K.mean(x[0], axis = [1, 2], keepdims = True)
    std = K.std(x[0], axis = [1, 2], keepdims = True) + 1e-7
    y = (x[0] - mean) / std

    #Reshape gamma and beta
    pool_shape = [-1, 1, 1, y.shape[-1]]
    g = tf.reshape(x[1], pool_shape) + 1.0
    b = tf.reshape(x[2], pool_shape)

    #Multiply by x[1] (GAMMA) and add x[2] (BETA)
    return y * g + b

def fade_block(x, block_num):
    #Inputs: [small-res (a), big-res (1-a), alpha]
    sr = x[0]
    br = x[1]
    alpha = x[2]

    alpha = tf.reshape(alpha, [-1, 1, 1, 1])
    alpha = tf.clip(alpha - block_num, 0, 1)

    return (sr * alpha) + (br * (1 - alpha))

def crop_to_fit(x):

    height = x[1].shape[1]
    width = x[1].shape[2]

    return x[0][:, :height, :width, :]


#Blocks
def g_block(inp, style, inoise, fil, u = True):

    if u:
        out = UpSampling2D()(inp)
    else:
        out = Activation('linear')(inp)

    gamma = Dense(fil)(style)
    beta = Dense(fil)(style)

    delta = Lambda(crop_to_fit)([inoise, out])
    delta = Dense(fil, kernel_initializer = 'zeros')(delta)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = add([out, delta])
    out = Lambda(AdaIN)([out, gamma, beta])
    out = LeakyReLU(0.2)(out)

    return out

def d_block(inp, fil, p = True):

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
    out = LeakyReLU(0.2)(out)

    if p:
        out = AveragePooling2D()(out)

    return out

class GAN(object):

    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001):

        #Models
        self.D = None
        self.S = None
        self.G = None

        self.GE = None
        self.SE = None

        self.DM = None
        self.AM = None

        #Config
        self.LR = lr
        self.steps = steps
        self.beta = 0.999

        #Init Models
        self.discriminator()
        self.generator()

        self.GMO = Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.9)
        self.DMO = Adam(lr = self.LR * 4, beta_1 = 0, beta_2 = 0.9)

        self.GE = model_from_json(self.G.to_json())
        self.GE.set_weights(self.G.get_weights())

        self.SE = model_from_json(self.S.to_json())
        self.SE.set_weights(self.S.get_weights())

    def discriminator(self):

        if self.D:
            return self.D

        inp = Input(shape = [im_size, im_size, 3])

        x = d_block(inp, 1 * cha)   #128
        x = d_block(x, 2 * cha)   #64
        x = d_block(x, 3 * cha)   #32
        x = d_block(x, 4 * cha)  #16
        x = d_block(x, 6 * cha)  #8
        x = d_block(x, 8 * cha)  #4
        x = d_block(x, 16 * cha, p = False)  #4

        x = Flatten()(x)

        x = Dense(16 * cha, kernel_initializer = 'he_normal')(x)
        x = LeakyReLU(0.2)(x)

        x = Dense(1, kernel_initializer = 'he_normal')(x)

        self.D = Model(inputs = inp, outputs = x)

        return self.D

    def generator(self):

        if self.G:
            return self.G

        # === Style Mapping ===

        self.S = Sequential()

        self.S.add(Dense(512, input_shape = [latent_size]))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))


        # === Generator ===

        #Inputs
        inp_style = []

        for i in range(n_layers):
            inp_style.append(Input([512]))

        inp_noise = Input([im_size, im_size, 1])

        #Latent
        x = Lambda(lambda x: x[:, :128])(inp_style[0])

        #Actual Model
        x = Dense(4*4*4*cha, activation = 'relu', kernel_initializer = 'he_normal')(x)
        x = Reshape([4, 4, 4*cha])(x)
        x = g_block(x, inp_style[0], inp_noise, 16 * cha, u = False)  #4
        x = g_block(x, inp_style[1], inp_noise, 8 * cha)  #8
        x = g_block(x, inp_style[2], inp_noise, 6 * cha)  #16
        x = g_block(x, inp_style[3], inp_noise, 4 * cha)  #32
        x = g_block(x, inp_style[4], inp_noise, 3 * cha)   #64
        x = g_block(x, inp_style[5], inp_noise, 2 * cha)   #128
        x = g_block(x, inp_style[6], inp_noise, 1 * cha)   #256

        x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(x)

        self.G = Model(inputs = inp_style + [inp_noise], outputs = x)

        return self.G

    def GenModel(self):

        inp_style = []
        style = []

        for i in range(n_layers):
            inp_style.append(Input([latent_size]))
            style.append(self.S(inp_style[-1]))

        inp_noise = Input([im_size, im_size, 1])

        gf = self.G(style + [inp_noise])

        self.GM = Model(inputs = inp_style + [inp_noise], outputs = gf)

        return self.GM

    def GenModelA(self):

        inp_style = []
        style = []
        trunc = Input([1])

        for i in range(n_layers):
            inp_style.append(Input([latent_size]))
            style.append(self.SE(inp_style[-1]))
            style[-1] = Lambda(lambda x: x * trunc)(style[-1])

        inp_noise = Input([im_size, im_size, 1])

        gf = self.GE(style + [inp_noise])

        self.GMA = Model(inputs = inp_style + [inp_noise, trunc], outputs = gf)

        return self.GMA

    def EMA(self):

        for i in range(len(self.G.layers)):
            up_weight = self.G.layers[i].get_weights()
            old_weight = self.GE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.GE.layers[i].set_weights(new_weight)

        for i in range(len(self.S.layers)):
            up_weight = self.S.layers[i].get_weights()
            old_weight = self.SE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.SE.layers[i].set_weights(new_weight)

    def MAinit(self):
        self.GE.set_weights(self.G.get_weights())
        self.SE.set_weights(self.S.get_weights())






class StyleGAN(object):

    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001, silent = True):

        self.GAN = GAN(steps = steps, lr = lr, decay = decay)
        self.GAN.GenModel()
        self.GAN.GenModelA()

        self.GAN.G.summary()

        self.lastblip = time.clock()

        self.noise_level = 0

        self.im = dataGenerator(directory, im_size, flip = True)

        self.silent = silent

        #Train Generator to be in the middle, not all the way at real. Apparently works better??
        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones

        self.gp_weight = np.array([10.0] * BATCH_SIZE).astype('float32')

    def train(self):

        #Train Alternating
        if random() < mixed_prob:
            style = mixedList(BATCH_SIZE)
        else:
            style = noiseList(BATCH_SIZE)

        a, b, c = self.train_step(self.im.get_batch(BATCH_SIZE).astype('float32'), style, nImage(BATCH_SIZE), self.gp_weight)

        if self.GAN.steps % 10 == 0:
            self.GAN.EMA()

        if self.GAN.steps <= 10000 and self.GAN.steps % 1000 == 2:
            self.GAN.MAinit()

        new_weight = 5/(np.array(c) + 1e-7)
        self.gp_weight = self.gp_weight[0] * 0.9 + 0.1 * new_weight
        self.gp_weight = np.clip([self.gp_weight] * BATCH_SIZE, 0.01, 10000.0).astype('float32')


        #Print info
        if self.GAN.steps % 100 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D:", np.array(a))
            print("G:", np.array(b))
            print("GP:", self.gp_weight[0])

            s = round((time.clock() - self.lastblip), 4)
            self.lastblip = time.clock()

            steps_per_second = 100 / s
            steps_per_minute = steps_per_second * 60
            steps_per_hour = steps_per_minute * 60
            print("Steps/Second: " + str(round(steps_per_second, 2)))
            print("Steps/Hour: " + str(round(steps_per_hour)))

            min1k = floor(1000/steps_per_minute)
            sec1k = floor(1000/steps_per_second) % 60
            print("1k Steps: " + str(min1k) + ":" + str(sec1k))
            steps_left = 200000 - self.GAN.steps + 1e-7
            hours_left = steps_left // steps_per_hour
            minutes_left = (steps_left // steps_per_minute) % 60

            print("Til Completion: " + str(int(hours_left)) + "h" + str(int(minutes_left)) + "m")
            print()

            #Save Model
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 10000))
            if self.GAN.steps % 1000 == 0 or (self.GAN.steps % 100 == 0 and self.GAN.steps < 1000):
                self.evaluate(floor(self.GAN.steps / 1000))


        printProgressBar(self.GAN.steps % 100, 99, decimals = 0)

        self.GAN.steps = self.GAN.steps + 1

    @tf.function
    def train_step(self, images, style, noise, gp_weights):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = self.GAN.GM(style + [noise], training=True)

          real_output = self.GAN.D(images, training=True)
          fake_output = self.GAN.D(generated_images, training=True)

          gen_loss = K.mean(fake_output)
          divergence = K.mean(K.relu(1 + real_output) + K.relu(1 - fake_output))
          disc_loss = divergence + gradient_penalty(images, real_output, gp_weights)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.GAN.GM.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.GAN.D.trainable_variables)

        self.GAN.GMO.apply_gradients(zip(gradients_of_generator, self.GAN.GM.trainable_variables))
        self.GAN.DMO.apply_gradients(zip(gradients_of_discriminator, self.GAN.D.trainable_variables))

        return disc_loss, gen_loss, divergence

    def evaluate(self, num = 0, trunc = 1.0):

        n1 = noiseList(64)
        n2 = nImage(64)
        trunc = np.ones([64, 1]) * trunc


        generated_images = self.GAN.GM.predict(n1 + [n2], batch_size = BATCH_SIZE)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1*255))

        x.save("Results/i"+str(num)+".png")

        # Moving Average

        generated_images = self.GAN.GMA.predict(n1 + [n2, trunc], batch_size = BATCH_SIZE)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)

        x = Image.fromarray(np.uint8(c1*255))

        x.save("Results/i"+str(num)+"-ema.png")

        #Mixing Regularities
        nn = noise(8)
        n1 = np.tile(nn, (8, 1))
        n2 = np.repeat(nn, 8, axis = 0)
        tt = int(n_layers / 2)

        p1 = [n1] * tt
        p2 = [n2] * (n_layers - tt)

        latent = p1 + [] + p2

        generated_images = self.GAN.GMA.predict(latent + [nImage(64), trunc], batch_size = BATCH_SIZE)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 0))

        c1 = np.concatenate(r, axis = 1)
        c1 = np.clip(c1, 0.0, 1.0)

        x = Image.fromarray(np.uint8(c1*255))

        x.save("Results/i"+str(num)+"-mr.png")

    def saveModel(self, model, name, num):
        json = model.to_json()
        with open("Models/"+name+".json", "w") as json_file:
            json_file.write(json)

        model.save_weights("Models/"+name+"_"+str(num)+".h5")

    def loadModel(self, name, num):

        file = open("Models/"+name+".json", 'r')
        json = file.read()
        file.close()

        mod = model_from_json(json)
        mod.load_weights("Models/"+name+"_"+str(num)+".h5")

        return mod

    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.S, "sty", num)
        self.saveModel(self.GAN.G, "gen", num)
        self.saveModel(self.GAN.D, "dis", num)

        self.saveModel(self.GAN.GE, "genMA", num)
        self.saveModel(self.GAN.SE, "styMA", num)


    def load(self, num): #Load JSON and Weights from /Models/

        #Load Models
        self.GAN.D = self.loadModel("dis", num)
        self.GAN.S = self.loadModel("sty", num)
        self.GAN.G = self.loadModel("gen", num)

        self.GAN.GE = self.loadModel("genMA", num)
        self.GAN.SE = self.loadModel("styMA", num)

        self.GAN.GenModel()
        self.GAN.GenModelA()

    def createFrame(self, list1, list2, nIm, alpha = 0, fnum = 0):

        n1 = noiseList(list1[0].shape[0])

        n1 = [list1 * (1-alpha) + list2 * (alpha)] * n_layers

        im = self.GAN.GE.predict(n1 + [nIm], batch_size = BATCH_SIZE)

        r = []
        for i in range(0, 8*4, 4):
            r.append(np.concatenate(im[i : i+4], axis = 0))
        c = np.concatenate(r, axis = 1)
        c = np.clip(c, 0.0, 1.0)

        x = Image.fromarray(np.uint8(c*255), mode = 'RGB')
        x.save("Results/Frames/frame-"+str(fnum)+".png")

    def createWalk(self):

        iNoise = self.GAN.SE.predict(noise(32))
        noise1 = iNoise.copy()
        noise2 = self.GAN.SE.predict(noise(32))
        nIm = nImage(32)
        k = 0


        for round in range(30):
            for between in range(120):
                alpha = between / 120.0
                self.createFrame(noise1, noise2, nIm, alpha, k)
                k = k + 1

                print(round*120 + between, "frames.")

            noise1 = noise2.copy()
            noise2 = self.GAN.SE.predict(noiseList(32))
            if round >= 28:
                noise2 = iNoise.copy()









if __name__ == "__main__":
    model = StyleGAN(lr = 0.0001, silent = False)
    model.evaluate(0)

    while model.GAN.steps <= 1000001:
        model.train()
