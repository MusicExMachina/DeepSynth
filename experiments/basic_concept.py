import keras


from keras.layers import Dense, Input


WINDOW_SIZE = 100

#MODEL
inp = Input(shape=(WINDOW_SIZE,))
h1 = Dense(WINDOW_SIZE//2)(inp)
out = Dense(2)(h1)

model = Model(inputs=inp,outputs=out)



#DATAGEN
def def_generate_blip(start,length,half_wavlen,clip_size):
    clip = np.zeros(clip_size)
    osc = 1
    for i in range(start,start+length):
        if i%(half_wavlen) == 0:
            osc *= -1
        clip[i] = osc
    return clip

print(def_generate_blip(200,600,20,1000))